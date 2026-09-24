import os
import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from functools import wraps

import discord
from discord.ext import commands
from mcp.server import Server
from mcp.types import Tool, TextContent, EmptyResult
from mcp.server.stdio import stdio_server
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("discord-mcp-server")

# Discord bot setup
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
if not DISCORD_TOKEN:
    raise ValueError("DISCORD_TOKEN environment variable is required")

# Default server ID (can be overridden with environment variable)
DEFAULT_SERVER_ID = os.getenv("DEFAULT_SERVER_ID")

# Presence: connect INVISIBLE only when explicitly opted in via MCP_DISCORD_CONNECT_INVISIBLE.
# Muraveynik's launchers set it so that merely having the gateway connection alive (any session that
# spawns this server) does NOT light the bot up — presence is then driven by the set_presence tool
# (the responder turns it online on start, invisible on stop). Other consumers of this SHARED server
# that don't set the flag keep discord.py's default (online), so this never silently darkens a peer bot.
_CONNECT_INVISIBLE = os.getenv("MCP_DISCORD_CONNECT_INVISIBLE", "").strip().lower() in ("1", "true", "yes", "on")
_initial_status = discord.Status.invisible if _CONNECT_INVISIBLE else discord.Status.online

intents = discord.Intents.default()
intents.message_content = True
intents.members = True
bot = commands.Bot(command_prefix="!", intents=intents, status=_initial_status)

# Desired presence, re-asserted on every (re-)connect (see on_ready/on_resumed). A full gateway
# re-IDENTIFY otherwise falls back to the IDENTIFY-time status, silently dropping a later set_presence;
# set_presence updates this dict so the re-assert keeps the dot correct while the client stays alive.
_desired_presence = {"status": _initial_status, "activity": None}


# ---- Voice messages (native) -------------------------------------------------------------------
# A Discord voice message is NOT a plain attachment: it needs the cloud-attachment upload flow
# (POST /channels/:id/attachments → PUT to the returned URL → POST /messages referencing it) plus
# flags=8192 (IS_VOICE_MESSAGE), duration_secs and a waveform. Sending the same file through
# multipart (discord.File) yields an ordinary audio attachment instead. Requires ffmpeg: any input
# audio is transcoded to the ogg/opus Discord clients expect, and the waveform is measured from PCM.
_VOICE_FLAG = 1 << 13          # 8192, IS_VOICE_MESSAGE
_WAVEFORM_MAX = 256            # Discord caps the waveform at 256 bytes


def _voice_encode(file_path: str):
    """Any audio file → (ogg/opus bytes, duration_secs, base64 waveform). Runs ffmpeg twice."""
    import base64
    import subprocess
    import tempfile
    from pathlib import Path as _P

    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as t:
        ogg_path = t.name
    try:
        subprocess.run(
            ["ffmpeg", "-i", file_path, "-ac", "1", "-ar", "48000",
             "-c:a", "libopus", "-b:a", "32k", "-y", ogg_path],
            capture_output=True, check=True,
        )
        ogg = _P(ogg_path).read_bytes()
    finally:
        _P(ogg_path).unlink(missing_ok=True)

    # Waveform: peak amplitude per bin over 8 kHz mono PCM (that is all the client renders).
    pcm = subprocess.run(
        ["ffmpeg", "-i", file_path, "-ac", "1", "-ar", "8000", "-f", "s16le", "-"],
        capture_output=True, check=True,
    ).stdout
    n_samples = len(pcm) // 2
    duration = round(n_samples / 8000, 2)
    bins = max(1, min(_WAVEFORM_MAX, int(duration * 10) or 1))
    step = max(1, n_samples // bins)
    peaks = []
    for i in range(0, n_samples, step):
        chunk = pcm[i * 2:(i + step) * 2]
        peak = 0
        for j in range(0, len(chunk) - 1, 2):
            v = int.from_bytes(chunk[j:j + 2], "little", signed=True)
            peak = max(peak, abs(v))
        peaks.append(min(255, peak * 255 // 32767))
        if len(peaks) >= bins:
            break
    return ogg, duration, base64.b64encode(bytes(peaks or [0])).decode()


async def _send_voice_message(channel_id: int, file_path: str, waveform_b64=None, duration=None):
    """Upload + post a native voice message; returns the message id."""
    import aiohttp

    ogg, dur, wf = _voice_encode(file_path)
    duration = duration or dur
    waveform_b64 = waveform_b64 or wf
    api = "https://discord.com/api/v10"
    headers = {"Authorization": f"Bot {DISCORD_TOKEN}"}
    filename = "voice-message.ogg"

    async with aiohttp.ClientSession(headers=headers) as sess:
        # 1. reserve an upload slot
        async with sess.post(f"{api}/channels/{channel_id}/attachments",
                             json={"files": [{"filename": filename, "file_size": len(ogg), "id": "0"}]}) as r:
            if r.status not in (200, 201):
                raise RuntimeError(f"attachment slot failed ({r.status}): {(await r.text())[:300]}")
            slot = (await r.json())["attachments"][0]

        # 2. PUT the bytes to the returned (pre-signed, unauthenticated) URL
        async with sess.put(slot["upload_url"], data=ogg,
                            headers={"Content-Type": "audio/ogg", "Authorization": ""}) as r:
            if r.status not in (200, 201):
                raise RuntimeError(f"upload failed ({r.status}): {(await r.text())[:300]}")

        # 3. post the message referencing the uploaded file
        payload = {
            "flags": _VOICE_FLAG,
            "attachments": [{
                "id": "0",
                "filename": filename,
                "uploaded_filename": slot["upload_filename"],
                "duration_secs": duration,
                "waveform": waveform_b64,
            }],
        }
        async with sess.post(f"{api}/channels/{channel_id}/messages", json=payload) as r:
            if r.status not in (200, 201):
                raise RuntimeError(f"send failed ({r.status}): {(await r.text())[:300]}")
            return (await r.json())["id"]


# ---- Custom (server) emojis --------------------------------------------------------------------
# Discord caps an emoji image at 256 KB (and renders it ~32-64 px), so anything drawn by an image
# model is far over the limit. Pillow downscales it here instead of failing the call — otherwise
# every freshly generated emoji would need a manual resize round-trip first.
_EMOJI_MAX_BYTES = 256 * 1024
_EMOJI_SIZES = (128, 96, 64)


async def _read_image_source(file_path=None, url=None) -> bytes:
    """Local path or image URL -> raw bytes. Shared by the emoji and sticker uploaders."""
    if file_path:
        from pathlib import Path as _P
        return _P(file_path).read_bytes()
    if url:
        import aiohttp
        async with aiohttp.ClientSession() as sess:
            async with sess.get(url) as r:
                if r.status != 200:
                    raise RuntimeError(f"image download failed ({r.status}): {url}")
                return await r.read()
    raise ValueError("either file_path or url is required")


async def _emoji_image_bytes(file_path=None, url=None) -> bytes:
    """Local path or image URL -> bytes that fit Discord's emoji size cap."""
    raw = await _read_image_source(file_path, url)
    return raw if len(raw) <= _EMOJI_MAX_BYTES else _shrink_emoji_image(raw)


def _shrink_emoji_image(raw: bytes) -> bytes:
    """Downscale an oversized image until it fits; animated GIFs keep their frames."""
    import io
    try:
        from PIL import Image, ImageSequence
    except ImportError:
        raise RuntimeError(
            f"image is {len(raw) // 1024} KB, over Discord's 256 KB emoji limit, and Pillow is not "
            "installed to shrink it — resize the file to 128x128 first"
        )

    src = Image.open(io.BytesIO(raw))
    animated = getattr(src, "is_animated", False)
    out = raw
    for size in _EMOJI_SIZES:
        buf = io.BytesIO()
        if animated:
            frames = []
            for frame in ImageSequence.Iterator(src):
                f = frame.convert("RGBA")
                f.thumbnail((size, size), Image.LANCZOS)
                frames.append(f)
            frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:],
                           loop=src.info.get("loop", 0), duration=src.info.get("duration", 100),
                           disposal=2, optimize=True)
        else:
            img = src.convert("RGBA")
            img.thumbnail((size, size), Image.LANCZOS)
            img.save(buf, format="PNG", optimize=True)
        out = buf.getvalue()
        if len(out) <= _EMOJI_MAX_BYTES:
            return out
    raise RuntimeError(f"could not shrink the image under 256 KB (smallest attempt: {len(out) // 1024} KB)")


def _find_emoji(emojis, ref: str):
    """Resolve an emoji by id, name, ':name:' or the '<:name:id>' form; None if absent."""
    import re
    ref = ref.strip()
    m = re.fullmatch(r"<a?:([^:]+):(\d+)>", ref)
    if m:
        ref = m.group(2)
    ref = ref.strip(":")
    if ref.isdigit():
        return next((e for e in emojis if str(e.id) == ref), None)
    return next((e for e in emojis if e.name.lower() == ref.lower()), None)


def _find_roles(guild_roles, refs):
    """Resolve role ids or role names; raises on the first unknown one (fail loud, not silently open)."""
    resolved = []
    for ref in refs:
        ref = str(ref).strip()
        role = next((r for r in guild_roles if str(r.id) == ref or r.name.lower() == ref.lower()), None)
        if not role:
            raise ValueError(f"role not found: {ref}")
        resolved.append(role)
    return resolved


def _emoji_line(emoji) -> str:
    usage = f"<a:{emoji.name}:{emoji.id}>" if emoji.animated else f"<:{emoji.name}:{emoji.id}>"
    limited = f", roles: {', '.join(r.name for r in emoji.roles)}" if emoji.roles else ""
    kind = "animated" if emoji.animated else "static"
    return f"{usage}  (name: {emoji.name}, ID: {emoji.id}, {kind}{limited})"


# ---- Stickers ----------------------------------------------------------------------------------
# Stickers are NOT big emojis: Discord demands exactly 320x320 (not "at most"), allows 512 KB, and
# requires a name + description + a unicode emoji tag. So the emoji fitter cannot be reused — it
# thumbnails to "no larger than", which leaves a non-square image the sticker endpoint rejects.
_STICKER_MAX_BYTES = 512 * 1024
_STICKER_SIDE = 320


def _fit_sticker_image(raw: bytes) -> bytes:
    """Any image -> exactly 320x320 PNG (APNG when animated) under Discord's 512 KB sticker cap."""
    import io
    try:
        from PIL import Image, ImageSequence
    except ImportError:
        raise RuntimeError(
            f"image is {len(raw) // 1024} KB / not 320x320, and Pillow is not installed to fit it — "
            "resize it to exactly 320x320 first"
        )

    def square(frame):
        """Pad to square THEN resize, so a non-square source is letterboxed instead of squashed."""
        f = frame.convert("RGBA")
        side = max(f.size)
        canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
        canvas.alpha_composite(f, ((side - f.width) // 2, (side - f.height) // 2))
        return canvas.resize((_STICKER_SIDE, _STICKER_SIDE), Image.LANCZOS)

    src = Image.open(io.BytesIO(raw))
    if len(raw) <= _STICKER_MAX_BYTES and src.size == (_STICKER_SIDE, _STICKER_SIDE):
        return raw                                   # already conformant — do not re-encode

    buf = io.BytesIO()
    if getattr(src, "is_animated", False):
        frames = [square(f) for f in ImageSequence.Iterator(src)]
        frames[0].save(buf, format="PNG", save_all=True, append_images=frames[1:],
                       loop=src.info.get("loop", 0), duration=src.info.get("duration", 100))
    else:
        square(src).save(buf, format="PNG", optimize=True)
    out = buf.getvalue()
    if len(out) > _STICKER_MAX_BYTES:
        # Last resort for photo-like art: 8-bit palette keeps 320x320 (mandatory) and drops the bytes.
        buf = io.BytesIO()
        square(src).convert("RGBA").quantize(colors=128, method=Image.Quantize.FASTOCTREE).save(
            buf, format="PNG", optimize=True)
        out = buf.getvalue()
    if len(out) > _STICKER_MAX_BYTES:
        raise RuntimeError(f"could not fit the sticker under 512 KB (smallest attempt: {len(out) // 1024} KB)")
    return out


def _find_sticker(stickers, ref: str):
    """Resolve a sticker by id or name; None if absent."""
    ref = ref.strip()
    if ref.isdigit():
        return next((s for s in stickers if str(s.id) == ref), None)
    return next((s for s in stickers if s.name.lower() == ref.lower()), None)


def _sticker_line(sticker) -> str:
    tag = f", tag: {sticker.emoji}" if sticker.emoji else ""
    desc = f" — {sticker.description}" if sticker.description else ""
    return f"{sticker.name} (ID: {sticker.id}, {sticker.format.name}{tag}){desc}"


# ---- Roles -------------------------------------------------------------------------------------
# Handing someone a role is the one Discord write that can hand out POWER, and it fails in ways the
# caller cannot guess from a bare "403 Forbidden": the role may be integration-managed (Discord
# refuses manual assignment outright, even for the server owner), or it may sit at/above the bot's
# own top role in the hierarchy. Both are checked BEFORE the API call and named in the error, and
# list_roles shows them up front so the caller sees what is assignable without trying.

# Permissions that make a role a privilege grant rather than a label. Surfaced so that handing out
# a cosmetic role is never confused with handing out staff powers.
_PRIVILEGED_PERMS = (
    "administrator", "manage_guild", "manage_roles", "manage_channels", "manage_webhooks",
    "manage_expressions", "ban_members", "kick_members", "moderate_members", "manage_messages",
    "manage_nicknames", "manage_events", "manage_threads", "mention_everyone",
)


def _privileged_perms(role) -> List[str]:
    """Privilege-granting permissions this role carries; empty for a purely cosmetic role."""
    p = role.permissions
    if p.administrator:
        return ["administrator"]          # implies every other permission; listing them adds noise
    return [n for n in _PRIVILEGED_PERMS if getattr(p, n, False)]


def _resolve_role(guild, role_arg: str):
    """'@Птички' | 'птичк' | '123…' → Role or None (same id-or-name idiom as list_members' filter)."""
    role_arg = (role_arg or "").strip().lstrip("@")
    if not role_arg:
        return None
    if role_arg.isdigit():
        return guild.get_role(int(role_arg))
    low = role_arg.lower()
    exact = next((r for r in guild.roles if r.name.lower() == low), None)
    return exact or next((r for r in guild.roles if low in r.name.lower()), None)


def _unassignable_reason(role, me) -> Optional[str]:
    """Why we cannot give/take this role, or None when we can. Checked before the call so the caller
    gets the actual cause instead of a 403 they have to guess at."""
    if role.is_default():
        return "@everyone is not an assignable role — every member has it by definition."
    if role.managed:
        return (f"'{role.name}' is managed by an integration (a bot's own role, a booster or "
                "subscription role) — Discord does not allow assigning it manually to anyone.")
    if not (me.guild_permissions.manage_roles or me.guild_permissions.administrator):
        return "I don't have the 'Manage Roles' permission in this server."
    if role >= me.top_role:
        return (f"'{role.name}' (position {role.position}) is not below my own top role "
                f"'{me.top_role.name}' (position {me.top_role.position}) — Discord only lets a bot "
                "manage roles strictly beneath its highest one. Move my role above it in "
                "Server Settings → Roles.")
    return None


# --- Editing the role ITSELF (name, colour, icon, permissions, position) -------------------------
# A different kind of write from handing a role to someone: it changes what the role IS for everyone
# who already holds it, and deleting one is irreversible (Discord strips it from every member, no
# undo). Two failure modes appear here that add_role never meets, and both come back as a bare 403/
# 400 that names nothing, so both are checked BEFORE the call:
#   * escalation — Discord will not let a bot GRANT a permission it does not itself hold;
#   * role icons — they exist only on servers carrying ROLE_ICONS in guild.features (boost level 2).
_ROLE_ICON_MAX_BYTES = 256 * 1024          # same cap as a custom emoji
_ROLE_ICON_SIZES = (128, 96, 64)           # a role icon renders at ~20-24 px; 128 is already generous


def _parse_colour(value) -> discord.Colour:
    """'#E67E22' | 'e67e22' | 'rgb(230,126,34)' | 'blurple' | 'none' -> Colour ('none' = no colour).

    Discord has no "unset" colour: the default is the integer 0, which the client renders as the
    plain grey of an uncoloured role — so clearing a colour and never having one are the same thing.
    """
    s = str(value).strip()
    if s.lower() in ("", "none", "default", "clear", "нет", "убрать", "сбросить"):
        return discord.Colour.default()
    literal = s if (s.startswith("#") or s.lower().startswith(("0x", "rgb"))) else f"#{s}"
    try:
        return discord.Colour.from_str(literal)
    except (ValueError, IndexError):
        pass
    named = getattr(discord.Colour, s.lower().replace(" ", "_").replace("-", "_"), None)
    if callable(named):                     # discord.Colour.red(), .blurple(), .gold(), …
        try:
            got = named()
            if isinstance(got, discord.Colour):
                return got
        except TypeError:                   # a classmethod that wants arguments (from_str, from_rgb)
            pass
    raise ValueError(
        f"unrecognised colour '{value}' — pass a hex ('#E67E22' or 'E67E22'), 'rgb(230,126,34)', a "
        "Discord colour name (blurple, red, gold, teal, fuchsia, orange, …) or 'none' to clear it")


def _colour_str(role) -> str:
    return "none" if role.colour.value == 0 else str(role.colour)


def _parse_perm_names(names, what="permissions") -> List[str]:
    """['manage messages', 'Kick Members'] -> ['manage_messages', 'kick_members']; unknown -> ValueError.

    Names are Discord's own permission flags (discord.py's VALID_FLAGS), so an unknown one is a typo,
    not a silently-ignored no-op: a permission that quietly fails to apply is the worst outcome here.
    """
    valid = discord.Permissions.VALID_FLAGS
    out = []
    for raw in names:
        n = str(raw).strip().lower().lstrip("@").replace(" ", "_").replace("-", "_")
        if n not in valid:
            close = sorted(v for v in valid if n and (n in v or v in n))
            hint = f" Did you mean: {', '.join(close[:5])}?" if close else ""
            raise ValueError(f"unknown permission '{raw}' in {what}.{hint}")
        out.append(n)
    return out


def _permissions_from(spec) -> discord.Permissions:
    """Permission names (list or comma/space string), 'none', 'all', or a raw bitfield -> Permissions."""
    import re
    if isinstance(spec, str):
        s = spec.strip().lower()
        if s in ("", "none", "0", "нет"):
            return discord.Permissions.none()
        if s == "all":
            return discord.Permissions.all()
        if s.isdigit():
            return discord.Permissions(permissions=int(s))       # legacy bitfield form
        spec = [p for p in re.split(r"[,\s]+", s) if p]
    if isinstance(spec, int):
        return discord.Permissions(permissions=spec)
    return discord.Permissions(**{n: True for n in _parse_perm_names(spec)})


def _escalation_reason(new_perms: discord.Permissions, me) -> Optional[str]:
    """Permissions in `new_perms` that I don't hold myself — Discord refuses those with a bare 403."""
    if me.guild_permissions.administrator:
        return None
    missing = [n for n, on in new_perms if on and not getattr(me.guild_permissions, n, False)]
    if not missing:
        return None
    return ("Discord does not let a bot grant permissions it doesn't hold itself, and I'm missing: "
            + ", ".join(sorted(missing))
            + ". Give my own role those permissions first (Server Settings → Roles), or drop them "
              "from this request.")


def _uneditable_reason(role, me) -> Optional[str]:
    """Why we cannot edit/delete the role ITSELF, or None. Stricter than _unassignable_reason: the
    default role and integration-managed roles are off limits as objects, not just as grants."""
    if not (me.guild_permissions.manage_roles or me.guild_permissions.administrator):
        return "I don't have the 'Manage Roles' permission in this server."
    if role.is_default():
        return ("@everyone is the server-wide default role — it has no name, colour, icon or "
                "position of its own and cannot be deleted. Server-wide defaults live in "
                "Server Settings → Roles.")
    if role.managed:
        return (f"'{role.name}' is managed by an integration (a bot's own role, a booster or "
                "subscription role) — Discord owns it, so it cannot be renamed, recoloured or "
                "deleted here.")
    if role >= me.top_role:
        return (f"'{role.name}' (position {role.position}) is at or above my own top role "
                f"'{me.top_role.name}' (position {me.top_role.position}) — Discord only lets a bot "
                "change roles strictly beneath its highest one. Move my role above it in "
                "Server Settings → Roles.")
    return None


def _role_icons_available(guild) -> bool:
    """Role icons need the ROLE_ICONS guild feature (boost level 2); without it the API 400s."""
    return "ROLE_ICONS" in (getattr(guild, "features", None) or [])


def _fit_role_icon(raw: bytes) -> bytes:
    """Any image -> PNG under the 256 KB role-icon cap. Unlike emojis, role icons take PNG/JPEG only,
    so an animated source is flattened to its first frame instead of being kept as a GIF."""
    import io
    try:
        from PIL import Image
    except ImportError:
        if raw[:3] == b"GIF":
            raise RuntimeError("role icons must be PNG or JPEG (Discord rejects GIF) and Pillow is "
                               "not installed to convert it — convert the file first")
        if len(raw) > _ROLE_ICON_MAX_BYTES:
            raise RuntimeError(f"image is {len(raw) // 1024} KB, over Discord's 256 KB role-icon "
                               "limit, and Pillow is not installed to shrink it")
        return raw

    src = Image.open(io.BytesIO(raw))
    if (src.format in ("PNG", "JPEG") and not getattr(src, "is_animated", False)
            and len(raw) <= _ROLE_ICON_MAX_BYTES):
        return raw                                   # already conformant — do not re-encode
    out = raw
    for size in _ROLE_ICON_SIZES:
        buf = io.BytesIO()
        img = src.convert("RGBA")                    # frame 0 of an animated source
        img.thumbnail((size, size), Image.LANCZOS)
        img.save(buf, format="PNG", optimize=True)
        out = buf.getvalue()
        if len(out) <= _ROLE_ICON_MAX_BYTES:
            return out
    raise RuntimeError(f"could not shrink the icon under 256 KB (smallest attempt: {len(out) // 1024} KB)")


async def _role_icon_from(guild, emoji=None, file_path=None, url=None):
    """-> (display_icon, spoken description). A unicode emoji stays a string (Discord stores it as
    one); a CUSTOM server emoji is not accepted by the role endpoint, so its image is downloaded and
    uploaded as the icon instead — otherwise ':ежик:' would silently mean nothing here."""
    if file_path or url:
        return _fit_role_icon(await _read_image_source(file_path, url)), "custom image"
    ref = str(emoji).strip()
    custom = _find_emoji(await guild.fetch_emojis(), ref) if (":" in ref or ref.isdigit()) else None
    if custom:
        return _fit_role_icon(await custom.read()), f"image of :{custom.name}:"
    return ref, ref


async def _move_role(guild, role, position: int, reason):
    """Put the role at `position` and let DISCORD do the reshuffle.

    Role.edit(position=…) cannot be used here: discord.py computes the new ordering client-side and
    assumes every role has a DISTINCT position, while Discord happily gives two roles the same number
    (ordering them by id). On a tie the computed payload comes out one slot short and the role
    silently does not move — the call still "succeeds". The bulk endpoint has no such assumption.
    """
    await guild.edit_role_positions(positions={role: position}, reason=reason)


async def _refetch_role(guild, role_id):
    """Re-read a role straight from the API — the authority on where it ACTUALLY landed after a move."""
    try:
        return next((r for r in await guild.fetch_roles() if r.id == role_id), None)
    except discord.HTTPException:
        return None


async def _role_holder_count(guild, role, cap: int = 1000):
    """(holders, hit_cap) — how many members hold the role. Said out loud before an IRREVERSIBLE
    delete, because Discord strips the role from all of them silently. Best effort: needs the
    members intent and a member fetch, so any failure returns (None, False) instead of blocking."""
    try:
        n = 0
        seen = 0
        async for m in guild.fetch_members(limit=cap):
            seen += 1
            if any(r.id == role.id for r in m.roles):
                n += 1
        return n, seen >= cap
    except Exception:
        return None, False


def _role_line(role, me) -> str:
    """One list_roles line: what the role looks like, what it grants, whether I can hand it out."""
    blocked = _unassignable_reason(role, me)
    bits = [f"ID: {role.id}", f"position {role.position}", f"colour {_colour_str(role)}"]
    icon = role.display_icon
    if icon is not None:
        bits.append(f"icon {icon}" if isinstance(icon, str) else "icon: image")
    if role.managed:
        bits.append("integration-managed")
    if role.hoist:
        bits.append("shown separately")
    if role.mentionable:
        bits.append("mentionable")
    line = f"{'✋' if blocked else '✅'} {role.name} ({', '.join(bits)})"
    if blocked:
        # The full sentence lives in the error path; here just the short cause.
        if role.is_default():
            line += " [everyone has it]"
        elif role.managed:
            line += " [managed — never assignable]"
        elif role >= me.top_role:
            line += " [at/above my top role]"
        else:
            line += " [I lack Manage Roles]"
    perms = _privileged_perms(role)
    if perms:
        line += f"  ⚠ grants: {', '.join(perms)}"
    return line


# ---- Channels ----------------------------------------------------------------------------------
# Creating and renaming a channel is cheap and reversible; DELETING one is not — its messages go
# with it and Discord offers no undo. So every write here names what it is about to touch (kind,
# id, category) and refuses with the actual cause instead of the bare 403 Discord returns, while
# list_channels shows the whole tree with a ✅/✋ mark so the caller sees the ground before moving.
# NOTE: a fetched Guild has no channel cache (guild.get_channel is always None there) — every
# channel lookup goes through `await guild.fetch_channels()`, same as get_server_info does.

_CHANNEL_KINDS = {                    # discord.ChannelType.name -> what we call it in tool output
    "text": "text", "voice": "voice", "forum": "forum", "category": "category",
    "news": "announcement", "stage_voice": "stage", "media": "media",
    "news_thread": "thread", "public_thread": "thread", "private_thread": "thread",
}
_KIND_ICON = {"text": "#", "announcement": "📣", "voice": "🔊", "stage": "🎙", "forum": "🗂",
              "media": "🖼", "category": "▸", "thread": "🧵"}
# What create_channel can make. Everything else (stage, announcement, media) is rare enough that
# Егор makes it by hand — a wrong guess here would be a channel nobody asked for.
_CREATABLE = ("text", "voice", "forum")


def _channel_kind(channel) -> str:
    name = getattr(getattr(channel, "type", None), "name", None) or str(getattr(channel, "type", "?"))
    return _CHANNEL_KINDS.get(name, name)


def _resolve_channel(channels, ref: str):
    """'123…' | '#имя' | подстрока → channel or None (same id-or-name idiom as _resolve_role).

    `channels` is the list from guild.fetch_channels() — see the note above about the empty cache.
    """
    ref = (ref or "").strip().lstrip("#")
    if not ref:
        return None
    if ref.isdigit():
        return next((c for c in channels if c.id == int(ref)), None)
    low = ref.lower()
    exact = next((c for c in channels if c.name.lower() == low), None)
    return exact or next((c for c in channels if low in c.name.lower()), None)


def _is_private(channel) -> bool:
    """True when @everyone cannot see the channel (an explicit view_channel=False overwrite)."""
    try:
        ow = channel.overwrites_for(channel.guild.default_role)
        return ow.view_channel is False
    except Exception:
        return False


def _unmanageable_reason(channel, me) -> Optional[str]:
    """Why we cannot create/rename/move/delete inside this channel, or None when we can.

    Checked before the call so the caller gets the cause instead of a 403 to guess at. A channel
    overwrite can take Manage Channels away inside one channel even when the bot has it server-wide,
    so the per-channel view (permissions_for) is the authority and the guild-wide one only explains.
    """
    try:
        perms = channel.permissions_for(me)
    except Exception:                       # no member/overwrite data → fall back to guild-wide
        perms = me.guild_permissions
    if perms.administrator or perms.manage_channels:
        return None
    if not (me.guild_permissions.manage_channels or me.guild_permissions.administrator):
        return "I don't have the 'Manage Channels' permission in this server."
    return (f"my permissions are overridden inside '{channel.name}' — this channel denies me "
            "'Manage Channels', so only a server admin can change it "
            "(Channel Settings → Permissions).")


def _protected_reason(channel, guild) -> Optional[str]:
    """Channels Discord itself refuses to delete — named up front instead of coming back as a 403."""
    for attr, what in (("rules_channel", "the rules channel"),
                       ("public_updates_channel", "the moderator-updates channel")):
        c = getattr(guild, attr, None)
        if c is not None and c.id == channel.id:
            return (f"'{channel.name}' is {what} of this Community server — Discord does not allow "
                    "deleting it while Community is enabled (Server Settings → Community).")
    return None


def _channel_line(channel, me, indent="  ") -> str:
    """One tree line: icon, name, kind, id, 🔒 when private, ✅/✋ manageability with its cause."""
    kind = _channel_kind(channel)
    blocked = _unmanageable_reason(channel, me)
    mark = "✋ " + blocked if blocked else "✅"
    lock = " 🔒 private" if _is_private(channel) else ""
    return f"{indent}{_KIND_ICON.get(kind, '·')} {channel.name} ({kind}, ID: {channel.id}){lock} {mark}"


# Initialize MCP server
app = Server("discord-server")

# Store Discord client reference
discord_client = None

# Per-channel continuous "typing…" loops — kept alive until a message is sent or explicitly stopped.
_typing_tasks = {}          # channel_id -> asyncio.Task
_TYPING_INTERVAL = 8        # re-trigger before the ~10s indicator fades
_TYPING_MAX_SECONDS = 300   # safety cap: never loop forever


async def _typing_loop(channel_id: int):
    elapsed = 0
    try:
        while elapsed < _TYPING_MAX_SECONDS:
            await discord_client.http.send_typing(channel_id)
            await asyncio.sleep(_TYPING_INTERVAL)
            elapsed += _TYPING_INTERVAL
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.warning(f"typing loop for {channel_id} stopped: {e}")
    finally:
        _typing_tasks.pop(channel_id, None)


def _stop_typing(channel_id: int):
    t = _typing_tasks.pop(channel_id, None)
    if t and not t.done():
        t.cancel()


# Last message id this bot sent per channel (the thread's parent too). A late start_typing —
# issued after the reply already went out — must not light the indicator back up: that is how
# "typing…" used to hang for 10-20 s under a finished answer.
_last_sent = {}             # channel_id -> message snowflake


def _note_sent(channel, message_id: int | None = None):
    """A reply (message/file/voice/sticker/reaction) went out: stop typing here AND in the parent
    channel when this is a thread — answers often land in a thread opened on the pinged message."""
    ids = [channel.id]
    parent = getattr(channel, "parent_id", None)
    if parent:
        ids.append(parent)
    for cid in ids:
        _stop_typing(cid)
        if message_id:
            _last_sent[cid] = max(_last_sent.get(cid, 0), int(message_id))


async def _reassert_presence(reason: str):
    try:
        await bot.change_presence(**_desired_presence)
    except Exception as e:
        logger.warning(f"could not re-assert presence on {reason}: {e}")


@bot.event
async def on_ready():
    global discord_client
    discord_client = bot
    # on_ready fires on every READY (including reconnect / re-IDENTIFY), where the gateway would
    # otherwise re-send the IDENTIFY-time status and lose a later set_presence — so re-assert here.
    await _reassert_presence("ready")
    logger.info(f"Logged in as {bot.user.name}")


@bot.event
async def on_resumed():
    # A RESUME normally preserves live presence; re-assert defensively in case it was reset.
    await _reassert_presence("resume")

# Helper function to ensure Discord client is ready
def require_discord_client(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        if not discord_client:
            raise RuntimeError("Discord client not ready")
        return await func(*args, **kwargs)
    return wrapper

@app.list_tools()
async def list_tools() -> List[Tool]:
    """List available Discord tools."""
    return [
        # Server Information Tools
        Tool(
            name="get_server_info",
            description="Get information about a Discord server",
            inputSchema={
                "type": "object",
                "properties": {
                    "server_id": {
                        "type": "string",
                        "description": "Discord server (guild) ID. If not provided, the default server ID will be used."
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="list_members",
            description="Get a list of members in a server, each with their role names and avatar URL. Optionally filter to a single role (`role`) or to members who can view a channel (`channel_id`) — e.g. to collect everyone in a specific channel or with a specific role.",
            inputSchema={
                "type": "object",
                "properties": {
                    "server_id": {
                        "type": "string",
                        "description": "Discord server (guild) ID. If not provided, the default server ID will be used."
                    },
                    "limit": {
                        "type": "number",
                        "description": "Maximum number of members to fetch",
                        "minimum": 1,
                        "maximum": 1000
                    },
                    "role": {
                        "type": "string",
                        "description": "Optional: keep only members who have this role. Accepts a role ID or a role name (case-insensitive, e.g. 'Старички')."
                    },
                    "channel_id": {
                        "type": "string",
                        "description": "Optional: keep only members who can view this channel (i.e. who are 'in' that channel)."
                    }
                },
                "required": []
            }
        ),

        # Role Management Tools
        Tool(
            name="add_role",
            description=(
                "Give a member a role. `role` takes the role NAME or its ID — no need to look the ID "
                "up first. Refuses with the real reason (integration-managed role, role above my own "
                "top role, missing Manage Roles) instead of a bare 403, and says so harmlessly when "
                "the member already has the role. Warns when the role grants staff powers. "
                "Use list_roles to see what is assignable."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "server_id": {
                        "type": "string",
                        "description": "Discord server ID. If not provided, the default server ID will be used."
                    },
                    "user_id": {
                        "type": "string",
                        "description": "ID of the member to give the role to"
                    },
                    "role": {
                        "type": "string",
                        "description": "Role name (e.g. 'Птички', case-insensitive, '@' optional) or role ID"
                    },
                    "role_id": {
                        "type": "string",
                        "description": "Deprecated alias for `role` (ID only). Prefer `role`."
                    },
                    "reason": {
                        "type": "string",
                        "description": "Audit-log reason shown in the server's audit log (optional)"
                    }
                },
                "required": ["user_id"]
            }
        ),
        Tool(
            name="remove_role",
            description=(
                "Take a role away from a member. `role` takes the role NAME or its ID. Same explicit "
                "refusals as add_role (managed role, hierarchy, missing permission), and says so "
                "harmlessly when the member doesn't have the role in the first place."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "server_id": {
                        "type": "string",
                        "description": "Discord server ID. If not provided, the default server ID will be used."
                    },
                    "user_id": {
                        "type": "string",
                        "description": "ID of the member to take the role from"
                    },
                    "role": {
                        "type": "string",
                        "description": "Role name (e.g. 'Птички', case-insensitive, '@' optional) or role ID"
                    },
                    "role_id": {
                        "type": "string",
                        "description": "Deprecated alias for `role` (ID only). Prefer `role`."
                    },
                    "reason": {
                        "type": "string",
                        "description": "Audit-log reason shown in the server's audit log (optional)"
                    }
                },
                "required": ["user_id"]
            }
        ),

        # Channel Management Tools
        Tool(
            name="list_channels",
            description="List the server's channels as a tree (categories, then the channels inside them) with each channel's kind, ID, whether it is private, and whether I can manage it. Unlike get_server_info this also shows forum channels and says up front what I can and cannot change.",
            inputSchema={
                "type": "object",
                "properties": {
                    "server_id": {
                        "type": "string",
                        "description": "Discord server ID. If not provided, the default server ID will be used."
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="create_channel",
            description="Create a channel: text (default), voice or forum. Requires Manage Channels.",
            inputSchema={
                "type": "object",
                "properties": {
                    "server_id": {
                        "type": "string",
                        "description": "Discord server ID. If not provided, the default server ID will be used."
                    },
                    "name": {
                        "type": "string",
                        "description": "Channel name. Discord lowercases text/forum names and turns spaces into dashes — the tool reports the name it actually got."
                    },
                    "type": {
                        "type": "string",
                        "description": "Kind of channel to create (default: text)",
                        "enum": ["text", "voice", "forum"]
                    },
                    "category_id": {
                        "type": "string",
                        "description": "Optional category ID to place the channel in"
                    },
                    "topic": {
                        "type": "string",
                        "description": "Optional channel topic (text and forum channels only)"
                    },
                    "private": {
                        "type": "boolean",
                        "description": "If true, @everyone cannot see the channel (default: false). Open it up for a specific role afterwards with set_channel_permissions."
                    },
                    "reason": {
                        "type": "string",
                        "description": "Optional audit-log reason (say who asked)"
                    }
                },
                "required": ["name"]
            }
        ),
        Tool(
            name="delete_channel",
            description="Delete a channel PERMANENTLY, together with its messages — Discord has no undo. Deleting a category does not delete the channels inside it; they just lose their category. Requires Manage Channels.",
            inputSchema={
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "string",
                        "description": "ID of channel to delete"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for deletion"
                    }
                },
                "required": ["channel_id"]
            }
        ),
        Tool(
            name="create_thread",
            description="Create a new thread in a text channel",
            inputSchema={
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "string",
                        "description": "ID of the text channel to create thread in"
                    },
                    "name": {
                        "type": "string",
                        "description": "Name of the thread"
                    },
                    "message_id": {
                        "type": "string",
                        "description": "Optional message ID to start the thread from. If not provided, creates a public thread."
                    },
                    "auto_archive_duration": {
                        "type": "number",
                        "description": "Duration in minutes before the thread will archive. Must be one of: 60, 1440, 4320, 10080",
                        "enum": [60, 1440, 4320, 10080]
                    }
                },
                "required": ["channel_id", "name"]
            }
        ),
        Tool(
            name="create_forum_post",
            description="Create a new post in a forum channel (a thread with its initial message, optionally with a file attached)",
            inputSchema={
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "string",
                        "description": "ID of the forum channel to create the post in"
                    },
                    "name": {
                        "type": "string",
                        "description": "Title of the post (max 100 characters)"
                    },
                    "content": {
                        "type": "string",
                        "description": "Text of the post's initial message (max 2000 characters)"
                    },
                    "file_path": {
                        "type": "string",
                        "description": "Optional path to a file (e.g. an image) to attach to the initial message"
                    },
                    "auto_archive_duration": {
                        "type": "number",
                        "description": "Duration in minutes before the post's thread will archive. Must be one of: 60, 1440, 4320, 10080",
                        "enum": [60, 1440, 4320, 10080]
                    }
                },
                "required": ["channel_id", "name", "content"]
            }
        ),
        Tool(
            name="set_channel_permissions",
            description="Set permissions for a channel",
            inputSchema={
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "string",
                        "description": "ID of the channel to set permissions for"
                    },
                    "role_id": {
                        "type": "string",
                        "description": "ID of the role to set permissions for. Use 'everyone' for the @everyone role."
                    },
                    "allow_view": {
                        "type": "boolean",
                        "description": "Allow or deny viewing the channel"
                    },
                    "modify_everyone": {
                        "type": "boolean",
                        "description": "Also modify @everyone permissions"
                    },
                    "everyone_can_view": {
                        "type": "boolean",
                        "description": "If modify_everyone is true, controls whether @everyone can view the channel"
                    }
                },
                "required": ["channel_id", "role_id"]
            }
        ),
        Tool(
            name="move_channel",
            description="Move a channel to a different position or category",
            inputSchema={
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "string",
                        "description": "ID of the channel to move"
                    },
                    "position": {
                        "type": "number",
                        "description": "New position for the channel (0 = top)"
                    },
                    "category_id": {
                        "type": "string",
                        "description": "Optional: ID of the category to move the channel to"
                    },
                    "sync_permissions": {
                        "type": "boolean",
                        "description": "Whether to sync permissions with the new category (default: true)"
                    }
                },
                "required": ["channel_id"]
            }
        ),
        Tool(
            name="edit_channel",
            description="Change a channel's name, topic (description) or slowmode. Pass only what you want changed. Requires Manage Channels. Discord allows 2 such edits per channel per 10 minutes.",
            inputSchema={
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "string",
                        "description": "ID of the channel to edit"
                    },
                    "name": {
                        "type": "string",
                        "description": "New channel name (full name, including any leading emoji)"
                    },
                    "topic": {
                        "type": "string",
                        "description": "New channel topic / description (max 1024 chars). Empty string clears it. Text and forum channels only."
                    },
                    "slowmode_delay": {
                        "type": "number",
                        "description": "Seconds between messages per user, 0 turns slowmode off (max 21600)"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Optional audit-log reason"
                    }
                },
                "required": ["channel_id"]
            }
        ),
        Tool(
            name="create_category",
            description="Create a new category in a server",
            inputSchema={
                "type": "object",
                "properties": {
                    "server_id": {
                        "type": "string",
                        "description": "Discord server ID. If not provided, the default server ID will be used."
                    },
                    "name": {
                        "type": "string",
                        "description": "Category name"
                    },
                    "position": {
                        "type": "number",
                        "description": "Optional position of the category"
                    },
                    "restricted_role_id": {
                        "type": "string",
                        "description": "Optional: If provided, only this role can view the category and its channels"
                    },
                    "everyone_can_view": {
                        "type": "boolean",
                        "description": "Optional: Controls whether @everyone can view this category. Default is true."
                    },
                    "with_general_channel": {
                        "type": "boolean",
                        "description": "Optional: also create a '<name>-general' text channel inside (default: false — an empty category is what people usually mean)"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Optional audit-log reason (say who asked)"
                    }
                },
                "required": ["name"]
            }
        ),

        # Message Reaction Tools
        Tool(
            name="add_reaction",
            description="Add a reaction to a message",
            inputSchema={
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "string",
                        "description": "Channel containing the message"
                    },
                    "message_id": {
                        "type": "string",
                        "description": "Message to react to"
                    },
                    "emoji": {
                        "type": "string",
                        "description": "Emoji to react with (Unicode or custom emoji ID)"
                    }
                },
                "required": ["channel_id", "message_id", "emoji"]
            }
        ),
        Tool(
            name="start_typing",
            description="Start a CONTINUOUS 'Bot is typing…' indicator in a channel — re-triggered every ~8s so it stays visible the whole time you think. Auto-stops when you send a message/file/voice/sticker/reaction there or into a thread of that channel (or call stop_typing). Call right when you begin composing a reply.",
            inputSchema={
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "string",
                        "description": "Channel to show the typing indicator in"
                    },
                    "after_message_id": {
                        "type": "string",
                        "description": "Optional: the message you are answering. If this bot already replied after it (in this channel or in a thread opened on it), typing is NOT started — avoids a stale indicator under a finished reply."
                    }
                },
                "required": ["channel_id"]
            }
        ),
        Tool(
            name="stop_typing",
            description="Stop the continuous typing indicator started by start_typing (usually unnecessary — send_message stops it automatically).",
            inputSchema={
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "string",
                        "description": "Channel to stop the typing indicator in"
                    }
                },
                "required": ["channel_id"]
            }
        ),
        Tool(
            name="set_presence",
            description="Set the bot's account-wide presence (the online/offline dot). When the server is started with MCP_DISCORD_CONNECT_INVISIBLE, the bot connects invisible and stays dark until this is called. Use 'online' when the bot is actively present/listening and 'invisible' to make it appear offline (while staying connected). Optional activity_text shows a custom status line under the name. The last-set value is re-asserted across reconnects.",
            inputSchema={
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["online", "idle", "dnd", "invisible"],
                        "description": "Presence status: online (green), idle (yellow), dnd (red), invisible (appears offline while still connected)."
                    },
                    "activity_text": {
                        "type": "string",
                        "description": "Optional custom status text shown under the bot's name (omit or empty to clear it)."
                    }
                },
                "required": ["status"]
            }
        ),
        Tool(
            name="add_multiple_reactions",
            description="Add multiple reactions to a message",
            inputSchema={
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "string",
                        "description": "Channel containing the message"
                    },
                    "message_id": {
                        "type": "string",
                        "description": "Message to react to"
                    },
                    "emojis": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "description": "Emoji to react with (Unicode or custom emoji ID)"
                        },
                        "description": "List of emojis to add as reactions"
                    }
                },
                "required": ["channel_id", "message_id", "emojis"]
            }
        ),
        Tool(
            name="remove_reaction",
            description="Remove a reaction from a message",
            inputSchema={
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "string",
                        "description": "Channel containing the message"
                    },
                    "message_id": {
                        "type": "string",
                        "description": "Message to remove reaction from"
                    },
                    "emoji": {
                        "type": "string",
                        "description": "Emoji to remove (Unicode or custom emoji ID)"
                    }
                },
                "required": ["channel_id", "message_id", "emoji"]
            }
        ),
        Tool(
            name="send_message",
            description="Send a message to a specific channel",
            inputSchema={
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "string",
                        "description": "Discord channel ID"
                    },
                    "content": {
                        "type": "string",
                        "description": "Message content"
                    }
                },
                "required": ["channel_id", "content"]
            }
        ),
        Tool(
            name="edit_message",
            description=("Edit a message the bot itself posted earlier (Discord allows editing only own "
                         "messages). Replaces the whole content. Used to keep a channel-side digest of a "
                         "thread conversation up to date."),
            inputSchema={
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "string",
                        "description": "ID of the channel (or thread) the message lives in"
                    },
                    "message_id": {
                        "type": "string",
                        "description": "ID of the bot's own message to edit"
                    },
                    "content": {
                        "type": "string",
                        "description": "New full message content (replaces the old one)"
                    }
                },
                "required": ["channel_id", "message_id", "content"]
            }
        ),
        Tool(
            name="send_file",
            description="Send a file (image, document, etc.) to a Discord channel",
            inputSchema={
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "string",
                        "description": "Discord channel ID"
                    },
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to send"
                    },
                    "content": {
                        "type": "string",
                        "description": "Optional message text to accompany the file"
                    }
                },
                "required": ["channel_id", "file_path"]
            }
        ),
        Tool(
            name="send_voice_message",
            description=(
                "Send a NATIVE Discord voice message (round waveform bubble, not a file attachment). "
                "Takes any audio file (ogg/mp3/wav) and transcodes it; requires ffmpeg. "
                "Voice messages carry no text — use send_message separately if you need words."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "string",
                        "description": "Discord channel ID"
                    },
                    "file_path": {
                        "type": "string",
                        "description": "Path to the audio file to send as a voice message"
                    }
                },
                "required": ["channel_id", "file_path"]
            }
        ),
        Tool(
            name="download_attachment",
            description="Download a Discord attachment to a local file",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL of the attachment (from read_messages)"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Local path to save the file"
                    }
                },
                "required": ["url", "output_path"]
            }
        ),
        Tool(
            name="read_messages",
            description="Read recent messages from a channel",
            inputSchema={
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "string",
                        "description": "Discord channel ID"
                    },
                    "limit": {
                        "type": "number",
                        "description": "Number of messages to fetch (max 100)",
                        "minimum": 1,
                        "maximum": 100
                    }
                },
                "required": ["channel_id"]
            }
        ),
        Tool(
            name="get_user_info",
            description=(
                "Get information about a Discord user: avatar/banner CDN URLs (download the returned "
                "Avatar URL with download_attachment) plus, when they are a member of the server, "
                "their nickname, join date and the roles they currently hold — check this before "
                "changing anyone's roles with add_role/remove_role."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "Discord user ID"
                    },
                    "server_id": {
                        "type": "string",
                        "description": "Server to read their membership (nickname/roles) from. Defaults to the default server."
                    }
                },
                "required": ["user_id"]
            }
        ),
        Tool(
            name="moderate_message",
            description="Delete a message and optionally timeout the user",
            inputSchema={
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "string",
                        "description": "Channel ID containing the message"
                    },
                    "message_id": {
                        "type": "string",
                        "description": "ID of message to moderate"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for moderation (optional)"
                    },
                    "timeout_minutes": {
                        "type": "number",
                        "description": "Optional timeout duration in minutes",
                        "minimum": 0,
                        "maximum": 40320  # Max 4 weeks
                    }
                },
                "required": ["channel_id", "message_id"]
            }
        ),
        Tool(
            name="create_role",
            description=(
                "Create a role. Everything the Discord role editor offers is here: name, colour, "
                "icon, hoist, mentionable, permissions (by NAME — e.g. ['kick_members']) and a "
                "starting position. Refuses with the real reason (missing Manage Roles, a "
                "permission I don't hold myself and so cannot grant, no ROLE_ICONS on this server) "
                "instead of a bare 403/400."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "server_id": {
                        "type": "string",
                        "description": "Discord server ID. If not provided, the default server ID will be used."
                    },
                    "name": {
                        "type": "string",
                        "description": "Name of the role"
                    },
                    "color": {
                        "type": "string",
                        "description": "Colour: hex ('#E67E22' or 'E67E22'), 'rgb(230,126,34)', a Discord colour name ('blurple', 'red', 'gold', 'teal', 'fuchsia'…), or 'none' for no colour."
                    },
                    "hoist": {
                        "type": "boolean",
                        "description": "Show holders of this role in their own section of the member list (default: false)"
                    },
                    "mentionable": {
                        "type": "boolean",
                        "description": "Let anyone @-mention the role (default: false)"
                    },
                    "permissions": {
                        "type": ["array", "string"],
                        "items": {"type": "string"},
                        "description": "Permissions the role grants, as Discord flag NAMES: ['manage_messages', 'kick_members']. Also accepts 'none' (default — a purely cosmetic role), 'all', or a raw bitfield string. I can only grant permissions I hold myself."
                    },
                    "icon_emoji": {
                        "type": "string",
                        "description": "Role icon as an emoji: a unicode emoji ('🌱') or a custom server emoji (':name:' / '<:name:id>' — its image is uploaded, since Discord stores role icons as images). Needs ROLE_ICONS (server boost level 2)."
                    },
                    "icon_file_path": {
                        "type": "string",
                        "description": "Role icon from a local image file (PNG/JPEG; oversized images are downscaled). Needs ROLE_ICONS."
                    },
                    "icon_url": {
                        "type": "string",
                        "description": "Role icon downloaded from this image URL. Needs ROLE_ICONS."
                    },
                    "position": {
                        "type": "integer",
                        "description": "Optional starting position (1 = just above @everyone, higher = further up). Must stay below my own top role. Prefer `above`/`below` in edit_role for readability."
                    },
                    "reason": {
                        "type": "string",
                        "description": "Audit-log reason — say who asked and where."
                    }
                },
                "required": ["name"]
            }
        ),
        Tool(
            name="edit_role",
            description=(
                "Change an existing role: rename it, recolour it, set or clear its icon, hoist it, "
                "make it mentionable, rewrite its permissions (replace with `permissions`, or nudge "
                "with `grant_permissions` / `revoke_permissions`), and move it up or down "
                "(`position`, or `above`/`below` another role). `role` takes the role NAME or its "
                "ID. Refuses with the real reason — integration-managed role, role at/above my own "
                "top role, missing Manage Roles, a permission I cannot grant, no ROLE_ICONS on this "
                "server. Editing a role changes it for EVERYONE who already holds it."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "server_id": {
                        "type": "string",
                        "description": "Discord server ID. If not provided, the default server ID will be used."
                    },
                    "role": {
                        "type": "string",
                        "description": "Role to edit: name (case-insensitive, '@' optional) or ID"
                    },
                    "name": {
                        "type": "string",
                        "description": "New name"
                    },
                    "color": {
                        "type": "string",
                        "description": "New colour: hex ('#E67E22'), 'rgb(...)', a Discord colour name, or 'none' to clear it."
                    },
                    "hoist": {
                        "type": "boolean",
                        "description": "Show holders in their own section of the member list"
                    },
                    "mentionable": {
                        "type": "boolean",
                        "description": "Let anyone @-mention the role"
                    },
                    "permissions": {
                        "type": ["array", "string"],
                        "items": {"type": "string"},
                        "description": "REPLACE the role's whole permission set with these flag names (or 'none' / 'all' / a bitfield). Everything not listed is turned OFF — use grant_permissions/revoke_permissions to change only some."
                    },
                    "grant_permissions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Turn these permission flags ON, leaving the rest of the role's permissions as they are."
                    },
                    "revoke_permissions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Turn these permission flags OFF, leaving the rest as they are."
                    },
                    "icon_emoji": {
                        "type": "string",
                        "description": "New icon: unicode emoji ('🌱') or custom server emoji (':name:' / '<:name:id>', uploaded as an image). Needs ROLE_ICONS (boost level 2)."
                    },
                    "icon_file_path": {
                        "type": "string",
                        "description": "New icon from a local image file (PNG/JPEG; downscaled if oversized). Needs ROLE_ICONS."
                    },
                    "icon_url": {
                        "type": "string",
                        "description": "New icon downloaded from this image URL. Needs ROLE_ICONS."
                    },
                    "clear_icon": {
                        "type": "boolean",
                        "description": "Remove the role's icon entirely."
                    },
                    "position": {
                        "type": "integer",
                        "description": "Absolute position (1 = just above @everyone, higher = further up; list_roles prints each role's position). Must stay below my own top role."
                    },
                    "above": {
                        "type": "string",
                        "description": "Move this role directly ABOVE the named role (name or ID) — usually what 'сделай её выше X' means."
                    },
                    "below": {
                        "type": "string",
                        "description": "Move this role directly BELOW the named role (name or ID)."
                    },
                    "reason": {
                        "type": "string",
                        "description": "Audit-log reason — say who asked and where."
                    }
                },
                "required": ["role"]
            }
        ),
        Tool(
            name="delete_role",
            description=(
                "Delete a role. IRREVERSIBLE — Discord strips it from every member who holds it and "
                "there is no undo, so the answer says how many members hold it and what it granted. "
                "`role` takes the role NAME or its ID. Refuses with the real reason "
                "(integration-managed, at/above my own top role, missing Manage Roles)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "server_id": {
                        "type": "string",
                        "description": "Discord server ID. If not provided, the default server ID will be used."
                    },
                    "role": {
                        "type": "string",
                        "description": "Role to delete: name (case-insensitive, '@' optional) or ID"
                    },
                    "role_id": {
                        "type": "string",
                        "description": "Deprecated alias for `role` (ID only). Prefer `role`."
                    },
                    "reason": {
                        "type": "string",
                        "description": "Audit-log reason — say who asked and where."
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="list_roles",
            description=(
                "List the server's roles, highest first — name, ID, position, colour, icon, hoist/"
                "mentionable flags. Each line says whether I can actually assign it (hierarchy / "
                "integration-managed) and which staff permissions it grants, so you can see what "
                "add_role and edit_role will accept before calling them. Pass `role` to get ONE "
                "role in full, with every permission it grants spelled out. To see WHO holds a "
                "role, use list_members with its `role` filter."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "server_id": {
                        "type": "string",
                        "description": "Discord server ID. If not provided, the default server ID will be used."
                    },
                    "role": {
                        "type": "string",
                        "description": "Optional: show just this role (name or ID) in full detail — every permission it grants, its colour and icon."
                    }
                },
                "required": []
            }
        ),
        # Custom Emoji Tools
        Tool(
            name="list_emojis",
            description="List the server's custom emojis. Each line carries the exact string to type in a message or pass to add_reaction (`<:name:id>`, `<a:name:id>` when animated), plus any role restriction.",
            inputSchema={
                "type": "object",
                "properties": {
                    "server_id": {
                        "type": "string",
                        "description": "Discord server ID. If not provided, the default server ID will be used."
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="create_emoji",
            description="Upload a new custom emoji to the server from a local image (`file_path`) or an image URL (`url`) — e.g. one just generated. Oversized images are downscaled automatically to fit Discord's 256 KB limit. Returns the `<:name:id>` string to post or react with.",
            inputSchema={
                "type": "object",
                "properties": {
                    "server_id": {
                        "type": "string",
                        "description": "Discord server ID. If not provided, the default server ID will be used."
                    },
                    "name": {
                        "type": "string",
                        "description": "Emoji name (2-32 chars, letters/digits/underscore — this is what people type between colons)"
                    },
                    "file_path": {
                        "type": "string",
                        "description": "Path to a local image file (PNG/JPEG/GIF). Either this or url is required."
                    },
                    "url": {
                        "type": "string",
                        "description": "Image URL to upload from (e.g. a Discord attachment URL). Either this or file_path is required."
                    },
                    "roles": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional: restrict use of the emoji to these roles (IDs or names). Omit to let everyone use it."
                    },
                    "reason": {
                        "type": "string",
                        "description": "Audit-log reason (optional)"
                    }
                },
                "required": ["name"]
            }
        ),
        Tool(
            name="edit_emoji",
            description="Rename a custom emoji or change which roles may use it. Renaming changes the `<:name:id>` string, so messages that already used the old name render it with the new one.",
            inputSchema={
                "type": "object",
                "properties": {
                    "server_id": {
                        "type": "string",
                        "description": "Discord server ID. If not provided, the default server ID will be used."
                    },
                    "emoji": {
                        "type": "string",
                        "description": "The emoji to edit: its ID, its name, or the '<:name:id>' form."
                    },
                    "name": {
                        "type": "string",
                        "description": "New name (optional)"
                    },
                    "roles": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional: new role restriction (IDs or names). An empty array clears the restriction."
                    },
                    "reason": {
                        "type": "string",
                        "description": "Audit-log reason (optional)"
                    }
                },
                "required": ["emoji"]
            }
        ),
        Tool(
            name="delete_emoji",
            description="Delete a custom emoji from the server. Irreversible: messages and reactions that used it lose the image.",
            inputSchema={
                "type": "object",
                "properties": {
                    "server_id": {
                        "type": "string",
                        "description": "Discord server ID. If not provided, the default server ID will be used."
                    },
                    "emoji": {
                        "type": "string",
                        "description": "The emoji to delete: its ID, its name, or the '<:name:id>' form."
                    },
                    "reason": {
                        "type": "string",
                        "description": "Audit-log reason (optional)"
                    }
                },
                "required": ["emoji"]
            }
        ),
        # Sticker Tools
        Tool(
            name="list_stickers",
            description="List the server's custom stickers (name, ID, format, emoji tag, description) and how many sticker slots are used.",
            inputSchema={
                "type": "object",
                "properties": {
                    "server_id": {
                        "type": "string",
                        "description": "Discord server ID. If not provided, the default server ID will be used."
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="create_sticker",
            description="Upload a new custom sticker to the server from a local image (`file_path`) or an image URL (`url`). Discord requires exactly 320x320 and 512 KB — images are padded to square, resized and compressed automatically. Stickers are sent with send_sticker, not typed like emojis.",
            inputSchema={
                "type": "object",
                "properties": {
                    "server_id": {
                        "type": "string",
                        "description": "Discord server ID. If not provided, the default server ID will be used."
                    },
                    "name": {
                        "type": "string",
                        "description": "Sticker name (2-30 characters)"
                    },
                    "emoji": {
                        "type": "string",
                        "description": "One unicode emoji that tags the sticker's expression (e.g. '😺'); required by Discord and used by the sticker picker's search."
                    },
                    "description": {
                        "type": "string",
                        "description": "Sticker description, up to 100 characters (optional)"
                    },
                    "file_path": {
                        "type": "string",
                        "description": "Path to a local image (PNG; APNG for animated). Either this or url is required."
                    },
                    "url": {
                        "type": "string",
                        "description": "Image URL to upload from. Either this or file_path is required."
                    },
                    "reason": {
                        "type": "string",
                        "description": "Audit-log reason (optional)"
                    }
                },
                "required": ["name", "emoji"]
            }
        ),
        Tool(
            name="edit_sticker",
            description="Rename a custom sticker or change its description / emoji tag.",
            inputSchema={
                "type": "object",
                "properties": {
                    "server_id": {
                        "type": "string",
                        "description": "Discord server ID. If not provided, the default server ID will be used."
                    },
                    "sticker": {
                        "type": "string",
                        "description": "The sticker to edit: its ID or its name."
                    },
                    "name": {
                        "type": "string",
                        "description": "New name (optional)"
                    },
                    "description": {
                        "type": "string",
                        "description": "New description (optional)"
                    },
                    "emoji": {
                        "type": "string",
                        "description": "New unicode emoji tag (optional)"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Audit-log reason (optional)"
                    }
                },
                "required": ["sticker"]
            }
        ),
        Tool(
            name="delete_sticker",
            description="Delete a custom sticker from the server. Irreversible.",
            inputSchema={
                "type": "object",
                "properties": {
                    "server_id": {
                        "type": "string",
                        "description": "Discord server ID. If not provided, the default server ID will be used."
                    },
                    "sticker": {
                        "type": "string",
                        "description": "The sticker to delete: its ID or its name."
                    },
                    "reason": {
                        "type": "string",
                        "description": "Audit-log reason (optional)"
                    }
                },
                "required": ["sticker"]
            }
        ),
        Tool(
            name="send_sticker",
            description="Send a sticker to a channel. Stickers cannot be typed inside message text like emojis — they ride along with the message, so use this instead of send_message (optional `content` puts text in the same message).",
            inputSchema={
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "string",
                        "description": "Channel to send to"
                    },
                    "sticker": {
                        "type": "string",
                        "description": "Sticker to send: its ID, or its name on this server."
                    },
                    "content": {
                        "type": "string",
                        "description": "Optional text sent in the same message"
                    },
                    "server_id": {
                        "type": "string",
                        "description": "Server to resolve a sticker NAME against. If not provided, the default server ID will be used."
                    }
                },
                "required": ["channel_id", "sticker"]
            }
        ),
        Tool(
            name="kick_user",
            description="Kick a user from the server",
            inputSchema={
                "type": "object",
                "properties": {
                    "server_id": {
                        "type": "string",
                        "description": "Discord server ID. If not provided, the default server ID will be used."
                    },
                    "user_id": {
                        "type": "string",
                        "description": "ID of user to kick"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for kicking the user (optional)"
                    }
                },
                "required": ["user_id"]
            }
        ),
        Tool(
            name="ban_user",
            description="Ban a user from the server",
            inputSchema={
                "type": "object",
                "properties": {
                    "server_id": {
                        "type": "string",
                        "description": "Discord server ID. If not provided, the default server ID will be used."
                    },
                    "user_id": {
                        "type": "string",
                        "description": "ID of user to ban"
                    },
                    "delete_message_days": {
                        "type": "number",
                        "description": "Number of days worth of messages to delete (0-7)",
                        "minimum": 0,
                        "maximum": 7
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for banning the user (optional)"
                    }
                },
                "required": ["user_id"]
            }
        )
    ]


@app.call_tool()
@require_discord_client
async def call_tool(name: str, arguments: Any) -> List[TextContent]:
    """Handle Discord tool calls."""
    
    if name == "send_message":
        channel = await discord_client.fetch_channel(int(arguments["channel_id"]))
        message = await channel.send(arguments["content"])
        _note_sent(channel, message.id)              # reply sent → stop the typing loop (+ thread parent)
        return [TextContent(
            type="text",
            text=f"Message sent successfully. Message ID: {message.id}"
        )]

    elif name == "edit_message":
        # Own messages only — the Discord API refuses to edit anyone else's content, so a
        # Forbidden here almost always means "that message is not ours" (not a perms gap).
        channel = await discord_client.fetch_channel(int(arguments["channel_id"]))
        try:
            message = await channel.fetch_message(int(arguments["message_id"]))
            await message.edit(content=arguments["content"])
        except discord.NotFound:
            return [TextContent(
                type="text",
                text="Error: message not found (wrong channel_id/message_id, or it was deleted)"
            )]
        except discord.Forbidden:
            return [TextContent(
                type="text",
                text="Error: cannot edit that message — a bot may only edit messages it posted itself"
            )]
        except Exception as e:
            return [TextContent(type="text", text=f"Error editing message: {str(e)}")]
        return [TextContent(
            type="text",
            text=f"Message edited successfully. Message ID: {message.id}"
        )]

    elif name == "start_typing":
        cid = int(arguments["channel_id"])
        after = int(arguments.get("after_message_id") or 0)
        # the reply to that message already went out (here, or into a thread opened on it — its
        # id equals the message id) → a late start would hang "typing…" under a finished answer
        if after and (_last_sent.get(cid, 0) > after or after in _last_sent):
            return [TextContent(
                type="text",
                text=f"Typing skipped in channel {cid}: a reply after message {after} was already sent"
            )]
        _stop_typing(cid)                            # restart cleanly if one is already running
        _typing_tasks[cid] = asyncio.create_task(_typing_loop(cid))
        return [TextContent(
            type="text",
            text=f"Continuous typing indicator started in channel {cid}"
        )]

    elif name == "stop_typing":
        cid = int(arguments["channel_id"])
        _stop_typing(cid)
        return [TextContent(
            type="text",
            text=f"Typing indicator stopped in channel {cid}"
        )]

    elif name == "set_presence":
        status_map = {
            "online": discord.Status.online,
            "idle": discord.Status.idle,
            "dnd": discord.Status.dnd,
            "invisible": discord.Status.invisible,
        }
        status_key = arguments["status"]
        status = status_map.get(status_key)
        if status is None:
            raise ValueError(f"Unknown status: {status_key!r} (expected one of {list(status_map)})")
        activity_text = (arguments.get("activity_text") or "").strip()
        activity = discord.CustomActivity(name=activity_text) if activity_text else None
        # Remember it so on_ready/on_resumed can re-assert after a reconnect (re-IDENTIFY otherwise
        # reverts to the connect-time status and silently drops this).
        _desired_presence["status"] = status
        _desired_presence["activity"] = activity
        await discord_client.change_presence(status=status, activity=activity)
        suffix = f" with activity {activity_text!r}" if activity_text else ""
        return [TextContent(
            type="text",
            text=f"Presence set to {status_key}{suffix}"
        )]

    elif name == "send_file":
        channel = await discord_client.fetch_channel(int(arguments["channel_id"]))
        file_path = arguments["file_path"]
        content = arguments.get("content", "")

        file = discord.File(file_path)
        message = await channel.send(content=content if content else None, file=file)
        _note_sent(channel, message.id)
        return [TextContent(
            type="text",
            text=f"File sent successfully. Message ID: {message.id}"
        )]

    elif name == "send_voice_message":
        message_id = await _send_voice_message(int(arguments["channel_id"]), arguments["file_path"])
        _note_sent(await discord_client.fetch_channel(int(arguments["channel_id"])), message_id)
        return [TextContent(
            type="text",
            text=f"Voice message sent successfully. Message ID: {message_id}"
        )]

    elif name == "download_attachment":
        import aiohttp
        from pathlib import Path as PathLib
        url = arguments["url"]
        output_path = arguments["output_path"]

        # Ensure output directory exists
        PathLib(output_path).parent.mkdir(parents=True, exist_ok=True)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        with open(output_path, 'wb') as f:
                            f.write(await resp.read())
                        return [TextContent(
                            type="text",
                            text=f"Downloaded to {output_path}"
                        )]
                    else:
                        return [TextContent(
                            type="text",
                            text=f"Download failed: HTTP {resp.status}"
                        )]
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Download error: {str(e)}"
            )]

    elif name == "read_messages":
        channel = await discord_client.fetch_channel(int(arguments["channel_id"]))
        limit = min(int(arguments.get("limit", 10)), 100)
        fetch_users = arguments.get("fetch_reaction_users", False)  # Only fetch users if explicitly requested
        messages = []
        async for message in channel.history(limit=limit):
            reaction_data = []
            for reaction in message.reactions:
                emoji_str = str(reaction.emoji.name) if hasattr(reaction.emoji, 'name') and reaction.emoji.name else str(reaction.emoji.id) if hasattr(reaction.emoji, 'id') else str(reaction.emoji)
                reaction_info = {
                    "emoji": emoji_str,
                    "count": reaction.count
                }
                logger.error(f"Emoji: {emoji_str}")
                reaction_data.append(reaction_info)

            # Collect attachments
            attachments_data = []
            for attachment in message.attachments:
                attachments_data.append({
                    "id": str(attachment.id),
                    "filename": attachment.filename,
                    "url": attachment.url,
                    "content_type": attachment.content_type,
                    "size": attachment.size
                })

            messages.append({
                "id": str(message.id),
                "author": str(message.author),
                "content": message.content,
                "timestamp": message.created_at.isoformat(),
                "reactions": reaction_data,
                "attachments": attachments_data
            })
        message_texts = []
        for m in messages:
            reactions_text = ', '.join([f"{r['emoji']}({r['count']})" for r in m['reactions']]) if m['reactions'] else 'No reactions'
            attachments_text = ', '.join([f"{a['filename']} ({a['url']})" for a in m['attachments']]) if m['attachments'] else 'No attachments'
            message_texts.append(
                f"ID: {m['id']}\n{m['author']} ({m['timestamp']}): {m['content']}\n"
                f"Reactions: {reactions_text}\n"
                f"Attachments: {attachments_text}"
            )

        return [TextContent(
            type="text",
            text=f"Retrieved {len(messages)} messages:\n\n" + "\n\n".join(message_texts)
        )]

    elif name == "get_user_info":
        user = await discord_client.fetch_user(int(arguments["user_id"]))

        def _sized(asset, size=1024):
            # Asset.with_size sets ?size=; harmless on default avatars (CDN ignores it).
            try:
                return asset.with_size(size).url
            except Exception:
                return asset.url

        # display_avatar always resolves (custom avatar, or Discord's default placeholder).
        # `user.avatar` is None when the user has no custom avatar; `user.banner` needs a full
        # fetch_user (which we did) and is None when unset. URLs point at the public CDN, so
        # anyone can download them without a token once the hash is known.
        display_avatar_url = _sized(user.display_avatar)
        custom_avatar_url = _sized(user.avatar) if user.avatar else None
        banner_url = _sized(user.banner) if user.banner else None

        # Membership half: fetch_user above is the GLOBAL account and carries no nickname or roles.
        # Best-effort — a user who isn't in this server (or no server_id at all) simply has none.
        member_lines = []
        server_id = arguments.get("server_id", DEFAULT_SERVER_ID)
        if server_id:
            try:
                guild = await discord_client.fetch_guild(int(server_id))
                member = await guild.fetch_member(user.id)
                roles = [r for r in member.roles if not r.is_default()]   # drop @everyone
                roles.sort(key=lambda r: r.position, reverse=True)
                member_lines = [
                    f"Server: {guild.name}",
                    f"Nickname: {member.nick or '-'} (shown as: {member.display_name})",
                    f"Joined: {member.joined_at.isoformat() if member.joined_at else '-'}",
                    "Roles: " + (", ".join(f"{r.name} (ID: {r.id})" for r in roles) or "— (none)"),
                ]
            except discord.NotFound:
                member_lines = [f"Not a member of server {server_id}."]
            except Exception as e:
                member_lines = [f"(could not read server membership: {e})"]

        return [TextContent(
            type="text",
            text="User information:\n" +
                 f"Name: {user.name}#{user.discriminator}\n" +
                 f"Global name: {user.global_name or '-'}\n" +
                 f"ID: {user.id}\n" +
                 f"Bot: {user.bot}\n" +
                 f"Created: {user.created_at.isoformat()}\n" +
                 f"Avatar URL: {display_avatar_url}\n" +
                 f"Custom avatar: {custom_avatar_url or '(none — using Discord default)'}\n" +
                 f"Banner URL: {banner_url or '(none)'}" +
                 ("\n" + "\n".join(member_lines) if member_lines else "")
        )]

    elif name == "moderate_message":
        channel = await discord_client.fetch_channel(int(arguments["channel_id"]))
        message = await channel.fetch_message(int(arguments["message_id"]))
        
        # Delete the message - Message.delete() doesn't accept a reason parameter
        await message.delete()
        
        # Handle timeout if specified
        if "timeout_minutes" in arguments and arguments["timeout_minutes"] > 0:
            if isinstance(message.author, discord.Member):
                duration = discord.utils.utcnow() + datetime.timedelta(
                    minutes=arguments["timeout_minutes"]
                )
                await message.author.timeout(
                    duration,
                    reason=arguments.get("reason", "User timed out via MCP")
                )
                return [TextContent(
                    type="text",
                    text=f"Message deleted and user timed out for {arguments['timeout_minutes']} minutes."
                )]
        
        return [TextContent(
            type="text",
            text="Message deleted successfully."
        )]

    # Server Information Tools
    elif name == "get_server_info":
        server_id = arguments.get("server_id", DEFAULT_SERVER_ID)
        if not server_id:
            return [TextContent(
                type="text",
                text="Error: No server ID provided and no default server ID set. Set DEFAULT_SERVER_ID environment variable or provide server_id in the request."
            )]
            
        guild = await discord_client.fetch_guild(int(server_id))
        
        # Get basic guild info
        info = {
            "name": guild.name,
            "id": str(guild.id),
            "owner_id": str(guild.owner_id),
            "member_count": guild.member_count,
            "created_at": guild.created_at.isoformat(),
            "description": guild.description,
            "premium_tier": guild.premium_tier,
            "explicit_content_filter": str(guild.explicit_content_filter)
        }
        
        # Fetch all channels to get categories and channels
        channels = await guild.fetch_channels()
        
        # Separate categories and channels
        categories = []
        text_channels = []
        voice_channels = []
        
        for channel in channels:
            if isinstance(channel, discord.CategoryChannel):
                categories.append({
                    "id": str(channel.id),
                    "name": channel.name,
                    "position": channel.position
                })
            elif isinstance(channel, discord.TextChannel):
                text_channels.append({
                    "id": str(channel.id),
                    "name": channel.name,
                    "category_id": str(channel.category_id) if channel.category_id else "None",
                    "topic": channel.topic or "No topic"
                })
            elif isinstance(channel, discord.VoiceChannel):
                voice_channels.append({
                    "id": str(channel.id),
                    "name": channel.name,
                    "category_id": str(channel.category_id) if channel.category_id else "None"
                })
        
        # Sort channels by position
        categories.sort(key=lambda x: x["position"])
        
        # Create formatted output
        output = [f"Server Information:"]
        for k, v in info.items():
            output.append(f"{k}: {v}")
        
        output.append("\nCategories:")
        for cat in categories:
            output.append(f"  {cat['name']} (ID: {cat['id']})")
        
        output.append("\nText Channels:")
        for chan in text_channels:
            output.append(f"  #{chan['name']} (ID: {chan['id']}, Category: {chan['category_id']})")
            
        output.append("\nVoice Channels:")
        for chan in voice_channels:
            output.append(f"  🔊 {chan['name']} (ID: {chan['id']}, Category: {chan['category_id']})")
        
        return [TextContent(
            type="text",
            text="\n".join(output)
        )]

    elif name == "list_members":
        server_id = arguments.get("server_id", DEFAULT_SERVER_ID)
        if not server_id:
            return [TextContent(
                type="text",
                text="Error: No server ID provided and no default server ID set. Set DEFAULT_SERVER_ID environment variable or provide server_id in the request."
            )]
            
        guild = await discord_client.fetch_guild(int(server_id))
        limit = min(int(arguments.get("limit", 1000)), 1000)

        def _sized(asset, size=256):
            try:
                return asset.with_size(size).url
            except Exception:
                return asset.url

        # Optional role filter: match by role ID (all digits) or by name (case-insensitive).
        role_arg = (arguments.get("role") or "").strip().lstrip("@")
        target_role = None
        if role_arg:
            if role_arg.isdigit():
                target_role = guild.get_role(int(role_arg))
            else:
                low = role_arg.lower()
                target_role = next((r for r in guild.roles if r.name.lower() == low), None) \
                    or next((r for r in guild.roles if low in r.name.lower()), None)
            if target_role is None:
                return [TextContent(type="text", text=f"No role matching '{role_arg}' found in this server.")]

        # Optional channel filter: keep members who can view that channel ("in" the channel).
        target_channel = None
        channel_arg = (arguments.get("channel_id") or "").strip()
        if channel_arg:
            for ch in await guild.fetch_channels():
                if ch.id == int(channel_arg):
                    target_channel = ch
                    break
            if target_channel is None:
                return [TextContent(type="text", text=f"No channel with ID {channel_arg} found in this server.")]

        members = []
        async for member in guild.fetch_members(limit=limit):
            if target_role is not None and target_role not in member.roles:
                continue
            if target_channel is not None and not target_channel.permissions_for(member).view_channel:
                continue
            is_default_avatar = member.avatar is None and member.guild_avatar is None
            members.append({
                "id": str(member.id),
                "name": member.name,
                "display": member.display_name,       # guild nick → global name → username
                "bot": member.bot,
                "roles": [role.name for role in member.roles[1:]],  # Skip @everyone
                "avatar": _sized(member.display_avatar),            # guild avatar if set, else global/default
                "default_avatar": is_default_avatar,
            })

        scope = "Server Members"
        if target_role is not None:
            scope = f"Members with role '{target_role.name}'"
        if target_channel is not None:
            scope = (scope if target_role is not None else "Members") + f" in #{target_channel.name}"

        lines = [f"{scope} ({len(members)}):"]
        for m in members:
            tag = " [bot]" if m["bot"] else ""
            roles = ", ".join(m["roles"]) or "—"
            av = m["avatar"] + (" (default placeholder)" if m["default_avatar"] else "")
            lines.append(f"{m['display']}{tag} (@{m['name']}, ID: {m['id']}) | roles: {roles} | avatar: {av}")

        return [TextContent(type="text", text="\n".join(lines))]

    # Role Management Tools
    # add_role and remove_role are one handler: the guard rails (member exists, role resolves,
    # no-op, managed/hierarchy/permission) are identical and only the verb differs.
    elif name in ("add_role", "remove_role"):
        adding = name == "add_role"
        verb, prep = ("add", "to") if adding else ("remove", "from")
        server_id = arguments.get("server_id", DEFAULT_SERVER_ID)
        if not server_id:
            return [TextContent(
                type="text",
                text="Error: No server ID provided and no default server ID set. Set DEFAULT_SERVER_ID environment variable or provide server_id in the request."
            )]

        guild = await discord_client.fetch_guild(int(server_id))

        try:
            member = await guild.fetch_member(int(arguments["user_id"]))
        except discord.NotFound:
            return [TextContent(
                type="text",
                text=f"Error: nobody with ID {arguments['user_id']} is a member of this server (they may have left)."
            )]

        role_arg = str(arguments.get("role") or arguments.get("role_id") or "").strip()
        if not role_arg:
            return [TextContent(type="text", text="Error: which role? Pass `role` (name or ID).")]
        role = _resolve_role(guild, role_arg)
        if role is None:
            return [TextContent(
                type="text",
                text=f"Error: no role matching '{role_arg}' in this server. Call list_roles for the exact names and IDs."
            )]

        # No-op first: report it plainly instead of spending a write and an audit-log entry.
        has_role = any(r.id == role.id for r in member.roles)
        if adding and has_role:
            return [TextContent(
                type="text",
                text=f"{member.display_name} (ID: {member.id}) already has '{role.name}' — nothing to do."
            )]
        if not adding and not has_role:
            return [TextContent(
                type="text",
                text=f"{member.display_name} (ID: {member.id}) doesn't have '{role.name}' — nothing to do."
            )]

        me = await guild.fetch_member(discord_client.user.id)   # guild.me is None on a fetched guild
        blocked = _unassignable_reason(role, me)
        if blocked:
            return [TextContent(type="text", text=f"Cannot {verb} '{role.name}': {blocked}")]

        reason = arguments.get("reason") or f"Role {'added' if adding else 'removed'} via MCP"
        try:
            if adding:
                await member.add_roles(role, reason=reason)
            else:
                await member.remove_roles(role, reason=reason)
        except discord.Forbidden as e:
            return [TextContent(
                type="text",
                text=f"Discord refused to {verb} '{role.name}' {prep} {member.display_name}: {e}"
            )]
        except discord.HTTPException as e:
            return [TextContent(
                type="text",
                text=f"Discord error while trying to {verb} '{role.name}' {prep} {member.display_name}: {e}"
            )]

        # Say out loud when the role just granted staff powers — a role name alone doesn't show it.
        perms = _privileged_perms(role)
        note = f"\n⚠ This role grants: {', '.join(perms)}." if (adding and perms) else ""
        return [TextContent(
            type="text",
            text=(f"{'Added' if adding else 'Removed'} role '{role.name}' (ID: {role.id}) {prep} "
                  f"{member.display_name} (@{member.name}, ID: {member.id}).{note}")
        )]

    # Channel Management Tools
    elif name == "list_channels":
        server_id = arguments.get("server_id", DEFAULT_SERVER_ID)
        if not server_id:
            return [TextContent(
                type="text",
                text="Error: No server ID provided and no default server ID set. Set DEFAULT_SERVER_ID environment variable or provide server_id in the request."
            )]

        guild = await discord_client.fetch_guild(int(server_id))
        channels = await guild.fetch_channels()
        me = await guild.fetch_member(discord_client.user.id)   # guild.me is None on a fetched guild

        cats = sorted([c for c in channels if isinstance(c, discord.CategoryChannel)],
                      key=lambda c: c.position)
        rest = [c for c in channels if not isinstance(c, discord.CategoryChannel)]
        manageable = sum(1 for c in channels if _unmanageable_reason(c, me) is None)

        out = [f"{guild.name} (ID: {guild.id}) — {len(rest)} channels in {len(cats)} categories; "
               f"I can manage {manageable} of {len(channels)}."]

        def _block(title, items):
            out.append("")
            out.append(title)
            if not items:
                out.append("  (empty)")
            for ch in sorted(items, key=lambda c: (c.position, c.id)):
                out.append(_channel_line(ch, me))

        loose = [c for c in rest if c.category_id is None]
        if loose:
            _block("▸ (no category)", loose)
        for cat in cats:
            blocked = _unmanageable_reason(cat, me)
            mark = "✋ " + blocked if blocked else "✅"
            _block(f"▸ {cat.name} (category, ID: {cat.id}) {mark}",
                   [c for c in rest if c.category_id == cat.id])

        return [TextContent(type="text", text="\n".join(out))]

    elif name == "create_channel":
        server_id = arguments.get("server_id", DEFAULT_SERVER_ID)
        if not server_id:
            return [TextContent(
                type="text",
                text="Error: No server ID provided and no default server ID set. Set DEFAULT_SERVER_ID environment variable or provide server_id in the request."
            )]

        kind = str(arguments.get("type") or "text").strip().lower()
        if kind not in _CREATABLE:
            return [TextContent(
                type="text",
                text=f"Error: cannot create a '{kind}' channel — I make {', '.join(_CREATABLE)}."
            )]

        guild = await discord_client.fetch_guild(int(server_id))
        me = await guild.fetch_member(discord_client.user.id)
        if not (me.guild_permissions.manage_channels or me.guild_permissions.administrator):
            return [TextContent(
                type="text",
                text="Cannot create a channel: I don't have the 'Manage Channels' permission in this server."
            )]

        category = None
        if arguments.get("category_id"):
            channels = await guild.fetch_channels()
            category = _resolve_channel(channels, str(arguments["category_id"]))
            if category is None:
                return [TextContent(
                    type="text",
                    text=f"Error: no category with ID {arguments['category_id']} in this server. Call list_channels for the categories."
                )]
            if not isinstance(category, discord.CategoryChannel):
                return [TextContent(
                    type="text",
                    text=f"Error: '{category.name}' (ID: {category.id}) is a {_channel_kind(category)} channel, not a category."
                )]
            blocked = _unmanageable_reason(category, me)
            if blocked:
                return [TextContent(
                    type="text",
                    text=f"Cannot create a channel in '{category.name}': {blocked}"
                )]

        overwrites = None
        if arguments.get("private"):
            overwrites = {guild.default_role: discord.PermissionOverwrite(view_channel=False)}

        wanted = arguments["name"]
        kwargs = {"name": wanted, "category": category, "overwrites": overwrites,
                  "reason": arguments.get("reason") or "Channel created via MCP"}
        if arguments.get("topic") and kind in ("text", "forum"):
            kwargs["topic"] = arguments["topic"]
        maker = {"text": guild.create_text_channel, "voice": guild.create_voice_channel,
                 "forum": guild.create_forum}[kind]
        try:
            channel = await maker(**{k: v for k, v in kwargs.items() if v is not None})
        except discord.Forbidden as e:
            return [TextContent(type="text", text=f"Discord refused to create the channel: {e}")]
        except discord.HTTPException as e:
            return [TextContent(type="text", text=f"Discord error while creating the channel: {e}")]

        where = f" in category '{category.name}'" if category else " (no category)"
        lock = ", hidden from @everyone" if arguments.get("private") else ""
        # Discord silently normalises text/forum names (lowercase, spaces → dashes). Say so, or the
        # caller reports back a name that does not exist.
        renamed = ("" if channel.name == wanted else
                   f"\nDiscord normalised the name '{wanted}' to '{channel.name}'.")
        return [TextContent(
            type="text",
            text=(f"Created {kind} channel '{channel.name}' (ID: {channel.id}){where}{lock}."
                  f"{renamed}")
        )]

    elif name == "delete_channel":
        try:
            channel = await discord_client.fetch_channel(int(arguments["channel_id"]))
        except discord.NotFound:
            return [TextContent(
                type="text",
                text=f"Error: no channel with ID {arguments['channel_id']} (already deleted?)."
            )]
        guild = channel.guild
        me = await guild.fetch_member(discord_client.user.id)

        blocked = _unmanageable_reason(channel, me) or _protected_reason(channel, guild)
        if blocked:
            return [TextContent(type="text", text=f"Cannot delete '{channel.name}': {blocked}")]

        kind = _channel_kind(channel)
        # Deleting a category leaves its channels behind, uncategorised — name them, so nobody
        # believes a whole section just went away (or that it survived).
        orphans = ([c.name for c in guild.channels if getattr(c, "category_id", None) == channel.id]
                   if isinstance(channel, discord.CategoryChannel) else [])
        try:
            await channel.delete(reason=arguments.get("reason") or "Channel deleted via MCP")
        except discord.Forbidden as e:
            return [TextContent(type="text", text=f"Discord refused to delete '{channel.name}': {e}")]
        except discord.HTTPException as e:
            return [TextContent(type="text", text=f"Discord error while deleting '{channel.name}': {e}")]

        note = (f" The {len(orphans)} channels it held are still there, now without a category: "
                + ", ".join(orphans) + ".") if orphans else ""
        return [TextContent(
            type="text",
            text=f"Deleted {kind} channel '{channel.name}' (ID: {channel.id}) and its messages.{note}"
        )]
        
    elif name == "create_thread":
        try:
            channel = await discord_client.fetch_channel(int(arguments["channel_id"]))
            
            # Check if this is a text channel that can have threads
            if not isinstance(channel, discord.TextChannel):
                return [TextContent(
                    type="text",
                    text="Error: The specified channel is not a text channel. Only text channels can have threads. For forum channels use create_forum_post."
                )]
            
            thread_name = arguments["name"]
            auto_archive_duration = int(arguments.get("auto_archive_duration", 1440))  # Default: 1 day
            
            # Create thread based on whether a message_id is provided
            if "message_id" in arguments:
                # Create a thread from a message
                message = await channel.fetch_message(int(arguments["message_id"]))
                thread = await message.create_thread(
                    name=thread_name,
                    auto_archive_duration=auto_archive_duration
                )
                return [TextContent(
                    type="text",
                    text=f"Created thread #{thread.name} (ID: {thread.id}) from message in channel #{channel.name}"
                )]
            else:
                # Create a public thread not connected to a message
                thread = await channel.create_thread(
                    name=thread_name,
                    auto_archive_duration=auto_archive_duration,
                    type=discord.ChannelType.public_thread
                )
                return [TextContent(
                    type="text",
                    text=f"Created public thread #{thread.name} (ID: {thread.id}) in channel #{channel.name}"
                )]
        except discord.Forbidden:
            return [TextContent(
                type="text",
                text="Error: The bot does not have permissions to create threads in this channel."
            )]
        except discord.HTTPException as e:
            return [TextContent(
                type="text",
                text=f"Error creating thread: {str(e)}"
            )]

    elif name == "create_forum_post":
        try:
            channel = await discord_client.fetch_channel(int(arguments["channel_id"]))

            if not isinstance(channel, discord.ForumChannel):
                return [TextContent(
                    type="text",
                    text="Error: The specified channel is not a forum channel. For text channels use create_thread."
                )]

            kwargs = {
                "name": arguments["name"],
                "content": arguments["content"],
                "auto_archive_duration": int(arguments.get("auto_archive_duration", 10080)),
            }
            if arguments.get("file_path"):
                kwargs["file"] = discord.File(arguments["file_path"])

            thread, message = await channel.create_thread(**kwargs)
            return [TextContent(
                type="text",
                text=f"Created forum post '{thread.name}' (thread ID: {thread.id}, message ID: {message.id}) in #{channel.name}"
            )]
        except FileNotFoundError:
            return [TextContent(
                type="text",
                text=f"Error: File not found: {arguments.get('file_path')}"
            )]
        except discord.Forbidden:
            return [TextContent(
                type="text",
                text="Error: The bot does not have permissions to create posts in this forum channel."
            )]
        except discord.HTTPException as e:
            return [TextContent(
                type="text",
                text=f"Error creating forum post: {str(e)}"
            )]

    elif name == "set_channel_permissions":
        try:
            channel = await discord_client.fetch_channel(int(arguments["channel_id"]))
            role_id_str = arguments["role_id"]
            
            # Get guild
            guild = channel.guild
            if not guild:
                guild = await discord_client.fetch_guild(channel.guild_id)
                
            # For @everyone role, use the default role
            if role_id_str.lower() == "everyone":
                role = guild.default_role
                logger.info("Using @everyone role for permission settings")
            else:
                # Get the specified role by ID
                try:
                    role_id = int(role_id_str)
                    role = guild.get_role(role_id)
                    
                    # If the role isn't in the cache, look it up more directly
                    if not role:
                        roles = await guild.fetch_roles()
                        for r in roles:
                            if r.id == role_id:
                                role = r
                                break
                except ValueError:
                    # If role_id isn't a valid number and not 'everyone'
                    return [TextContent(
                        type="text",
                        text=f"Error: Invalid role ID '{role_id_str}'. Must be a valid role ID or 'everyone'."
                    )]
                        
            if not role:
                return [TextContent(
                    type="text",
                    text=f"Error: Role with ID {role_id_str} not found in the server."
                )]
            
            me = await guild.fetch_member(discord_client.user.id)
            blocked = _unmanageable_reason(channel, me)
            if blocked:
                return [TextContent(
                    type="text",
                    text=f"Cannot change permissions on '{channel.name}': {blocked}"
                )]

            # Set if we allow or deny viewing the channel for the role
            allow_view = arguments.get("allow_view", True)
            
            # Set up permissions for the specified role
            if allow_view:
                await channel.set_permissions(role, view_channel=True, send_messages=True, read_message_history=True)
                permission_state = "can now see"
            else:
                await channel.set_permissions(role, view_channel=False)
                permission_state = "can no longer see"
            
            # Handle @everyone permissions if specified and this isn't already the everyone role
            result_text = f"Updated permissions: Role {role.name} {permission_state} the channel #{channel.name}."
            
            if arguments.get("modify_everyone", False) and role != guild.default_role:
                everyone_role = guild.default_role
                everyone_can_view = arguments.get("everyone_can_view", False)
                
                if everyone_can_view:
                    await channel.set_permissions(everyone_role, view_channel=True)
                    result_text += " @everyone can now see the channel."
                else:
                    await channel.set_permissions(everyone_role, view_channel=False)
                    result_text += " @everyone can no longer see the channel."
            
            return [TextContent(
                type="text",
                text=result_text
            )]
        except discord.Forbidden:
            return [TextContent(
                type="text",
                text="Error: The bot does not have permissions to modify channel permissions."
            )]
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error setting channel permissions: {str(e)}"
            )]
        
    elif name == "move_channel":
        try:
            channel = await discord_client.fetch_channel(int(arguments["channel_id"]))
            me = await channel.guild.fetch_member(discord_client.user.id)
            blocked = _unmanageable_reason(channel, me)
            if blocked:
                return [TextContent(type="text", text=f"Cannot move '{channel.name}': {blocked}")]

            # Build edit kwargs
            edit_kwargs = {}

            # Handle position
            if "position" in arguments:
                edit_kwargs["position"] = int(arguments["position"])

            # Handle category
            if "category_id" in arguments:
                category = await discord_client.fetch_channel(int(arguments["category_id"]))
                if not isinstance(category, discord.CategoryChannel):
                    return [TextContent(
                        type="text",
                        text=f"Error: Channel {arguments['category_id']} is not a category"
                    )]
                edit_kwargs["category"] = category

                # Sync permissions by default when moving to a new category
                if arguments.get("sync_permissions", True):
                    edit_kwargs["sync_permissions"] = True

            # Apply changes
            await channel.edit(**edit_kwargs, reason="Channel moved via MCP")

            result_parts = [f"Moved channel #{channel.name}"]
            if "position" in edit_kwargs:
                result_parts.append(f"to position {edit_kwargs['position']}")
            if "category" in edit_kwargs:
                result_parts.append(f"into category {edit_kwargs['category'].name}")

            return [TextContent(
                type="text",
                text=" ".join(result_parts)
            )]
        except discord.Forbidden:
            return [TextContent(
                type="text",
                text="Error: Bot doesn't have permission to move this channel"
            )]
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error moving channel: {str(e)}"
            )]

    elif name == "edit_channel":
        try:
            channel = await discord_client.fetch_channel(int(arguments["channel_id"]))
        except discord.NotFound:
            return [TextContent(
                type="text",
                text=f"Error: no channel with ID {arguments['channel_id']}."
            )]

        edits, said = {}, []
        if arguments.get("name"):
            edits["name"] = arguments["name"]
            said.append(f"renamed to '{arguments['name']}'")
        if "topic" in arguments and arguments["topic"] is not None:
            if not isinstance(channel, (discord.TextChannel, discord.ForumChannel)):
                return [TextContent(
                    type="text",
                    text=f"Error: a {_channel_kind(channel)} channel has no topic to set."
                )]
            edits["topic"] = arguments["topic"]
            said.append("topic cleared" if arguments["topic"] == "" else "topic set")
        if arguments.get("slowmode_delay") is not None:
            edits["slowmode_delay"] = int(arguments["slowmode_delay"])
            said.append(f"slowmode {edits['slowmode_delay']}s")
        if not edits:
            return [TextContent(
                type="text",
                text="Error: nothing to change — pass name, topic or slowmode_delay."
            )]

        me = await channel.guild.fetch_member(discord_client.user.id)
        blocked = _unmanageable_reason(channel, me)
        if blocked:
            return [TextContent(type="text", text=f"Cannot edit '{channel.name}': {blocked}")]

        try:
            # Discord rate-limits channel name/topic edits to 2 / 10 min PER CHANNEL. On 429
            # discord.py does NOT raise — it silently sleeps retry_after (often hundreds of
            # seconds) and retries, hanging this tool call. Bound the wait: a normal edit is
            # < 1s; not done in time → we hit the limit, so cancel and say to try later.
            await asyncio.wait_for(
                channel.edit(reason=arguments.get("reason", "edited via MCP"), **edits),
                timeout=5)
        except (asyncio.TimeoutError, discord.RateLimited):
            return [TextContent(
                type="text",
                text="Rate-limited (Discord allows 2 name/topic edits per channel / 10 min). Try later."
            )]
        except discord.Forbidden as e:
            return [TextContent(
                type="text",
                text=f"Discord refused to edit '{channel.name}': {e}"
            )]
        except discord.HTTPException as e:
            return [TextContent(
                type="text",
                text=f"Discord error while editing '{channel.name}': {e}"
            )]
        return [TextContent(
            type="text",
            text=f"Channel {channel.id}: " + ", ".join(said) + "."
        )]

    elif name == "create_category":
        server_id = arguments.get("server_id", DEFAULT_SERVER_ID)
        if not server_id:
            return [TextContent(
                type="text",
                text="Error: No server ID provided and no default server ID set. Set DEFAULT_SERVER_ID environment variable or provide server_id in the request."
            )]
            
        guild = await discord_client.fetch_guild(int(server_id))
        position = arguments.get("position")
        
        try:
            # Determine visibility permissions
            everyone_can_view = arguments.get("everyone_can_view", True)
            restricted_role_id = arguments.get("restricted_role_id")
            
            # Set up permission overwrites
            overwrites = {}
            
            # Handle @everyone permissions
            overwrites[guild.default_role] = discord.PermissionOverwrite(view_channel=everyone_can_view)
            
            # Handle restricted role permissions if provided
            if restricted_role_id:
                role = guild.get_role(int(restricted_role_id))
                if not role:
                    roles = await guild.fetch_roles()
                    for r in roles:
                        if r.id == int(restricted_role_id):
                            role = r
                            break
                            
                if role:
                    overwrites[role] = discord.PermissionOverwrite(view_channel=True, send_messages=True, read_message_history=True)
                else:
                    logger.warning(f"Could not find role with ID {restricted_role_id}")
            
            # Create the category with specified permissions
            category = await guild.create_category(
                name=arguments["name"],
                overwrites=overwrites,
                position=position,
                reason=arguments.get("reason") or "Category created via MCP"
            )

            # A '<name>-general' channel used to appear here unasked. Now it is opt-in: "make a
            # category" means a category, and a channel nobody ordered is noise someone must delete.
            note = ""
            if arguments.get("with_general_channel"):
                text_channel = await guild.create_text_channel(
                    name=f"{arguments['name']}-general",
                    category=category,
                    reason="Default channel for new category"
                )
                note = f" with channel #{text_channel.name} (ID: {text_channel.id}) inside"

            if restricted_role_id and not everyone_can_view:
                who = ". Only the given role can view it."
            elif not everyone_can_view:
                who = ". @everyone cannot view it."
            else:
                who = "."

            return [TextContent(
                type="text",
                text=f"Created category '{category.name}' (ID: {category.id}){note}{who}"
            )]
        except discord.Forbidden as e:
            return [TextContent(
                type="text",
                text=f"Discord refused to create the category (do I have 'Manage Channels'?): {e}"
            )]
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error creating category: {str(e)}"
            )]

    # Message Reaction Tools
    elif name == "add_reaction":
        channel = await discord_client.fetch_channel(int(arguments["channel_id"]))
        message = await channel.fetch_message(int(arguments["message_id"]))
        await message.add_reaction(arguments["emoji"])
        _stop_typing(int(arguments["channel_id"]))   # a reaction is a reply too → stop the typing loop
        return [TextContent(
            type="text",
            text=f"Added reaction {arguments['emoji']} to message"
        )]

    elif name == "add_multiple_reactions":
        channel = await discord_client.fetch_channel(int(arguments["channel_id"]))
        message = await channel.fetch_message(int(arguments["message_id"]))
        for emoji in arguments["emojis"]:
            await message.add_reaction(emoji)
        _stop_typing(int(arguments["channel_id"]))   # a reaction is a reply too → stop the typing loop
        return [TextContent(
            type="text",
            text=f"Added reactions: {', '.join(arguments['emojis'])} to message"
        )]

    elif name == "remove_reaction":
        channel = await discord_client.fetch_channel(int(arguments["channel_id"]))
        message = await channel.fetch_message(int(arguments["message_id"]))
        await message.remove_reaction(arguments["emoji"], discord_client.user)
        return [TextContent(
            type="text",
            text=f"Removed reaction {arguments['emoji']} from message"
        )]
        
    # Role Management Tools
    # create_role / edit_role / delete_role / list_roles share their entire preamble — the guild, my
    # own member object (every hierarchy and escalation check needs it) and, for the three that name
    # one, the role itself. One branch keeps that in a single place, the same way add_role and
    # remove_role are one handler above.
    elif name in ("create_role", "edit_role", "delete_role", "list_roles"):
        server_id = arguments.get("server_id", DEFAULT_SERVER_ID)
        if not server_id:
            return [TextContent(
                type="text",
                text="Error: No server ID provided and no default server ID set. Set DEFAULT_SERVER_ID environment variable or provide server_id in the request."
            )]

        guild = await discord_client.fetch_guild(int(server_id))
        try:
            me = await guild.fetch_member(discord_client.user.id)
        except discord.HTTPException as e:
            return [TextContent(type="text", text=f"Could not read my own membership in {guild.name}: {e}")]

        # ---- list_roles: the whole ladder, or one role in full -------------------------------
        if name == "list_roles":
            try:
                roles = sorted(await guild.fetch_roles(), key=lambda r: r.position, reverse=True)
            except discord.HTTPException as e:
                return [TextContent(type="text", text=f"Error listing roles: {e}")]

            wanted = str(arguments.get("role") or "").strip()
            if wanted:
                role = _resolve_role(guild, wanted)
                if role is None:
                    return [TextContent(
                        type="text",
                        text=f"No role matching '{wanted}' in this server. Call list_roles without `role` for the full list."
                    )]
                grants = [n for n, on in role.permissions if on]
                lines = [
                    _role_line(role, me),
                    f"  Assignable: {_unassignable_reason(role, me) or 'yes'}",
                    f"  Editable:   {_uneditable_reason(role, me) or 'yes'}",
                    f"  Grants ({len(grants)}): " + (", ".join(sorted(grants)) or "— nothing (cosmetic)"),
                ]
                return [TextContent(type="text", text="\n".join(lines))]

            icons = ("role icons: available" if _role_icons_available(guild)
                     else "role icons: NOT available on this server (needs boost level 2)")
            lines = [
                f"Roles in {guild.name} ({len(roles)}), highest first. "
                f"My own top role: '{me.top_role.name}' (position {me.top_role.position}); {icons}.",
                "  ✅ I can add/remove it   ✋ I cannot (reason in brackets)   ⚠ grants staff powers",
            ]
            lines += [_role_line(r, me) for r in roles]
            return [TextContent(type="text", text="\n".join(lines))]

        # ---- create_role --------------------------------------------------------------------
        if name == "create_role":
            if not (me.guild_permissions.manage_roles or me.guild_permissions.administrator):
                return [TextContent(
                    type="text",
                    text="Cannot create a role: I don't have the 'Manage Roles' permission in this server."
                )]

            params = {"name": arguments["name"], "reason": arguments.get("reason", "Role created via MCP")}
            said = []
            try:
                if arguments.get("color") is not None:
                    colour = _parse_colour(arguments["color"])
                    params["colour"] = colour
                    said.append("no colour" if colour.value == 0 else f"colour {colour}")
                if arguments.get("hoist") is not None:
                    params["hoist"] = bool(arguments["hoist"])
                if arguments.get("mentionable") is not None:
                    params["mentionable"] = bool(arguments["mentionable"])
                if arguments.get("permissions") is not None:
                    perms = _permissions_from(arguments["permissions"])
                    blocked = _escalation_reason(perms, me)
                    if blocked:
                        return [TextContent(type="text", text=f"Cannot create '{arguments['name']}': {blocked}")]
                    params["permissions"] = perms
                if arguments.get("icon_emoji") or arguments.get("icon_file_path") or arguments.get("icon_url"):
                    if not _role_icons_available(guild):
                        return [TextContent(
                            type="text",
                            text=f"Cannot give '{arguments['name']}' an icon: {guild.name} doesn't have role icons "
                                 "(Discord unlocks them at server boost level 2). Everything else about the role still works."
                        )]
                    params["display_icon"], icon_said = await _role_icon_from(
                        guild, arguments.get("icon_emoji"), arguments.get("icon_file_path"), arguments.get("icon_url"))
                    said.append(f"icon {icon_said}")
            except (ValueError, RuntimeError) as e:
                return [TextContent(type="text", text=f"Error creating role: {e}")]

            try:
                role = await guild.create_role(**params)
            except discord.Forbidden as e:
                return [TextContent(type="text", text=f"Discord refused to create '{arguments['name']}': {e}")]
            except discord.HTTPException as e:
                return [TextContent(type="text", text=f"Discord error while creating '{arguments['name']}': {e}")]

            if arguments.get("position") is not None:
                pos = int(arguments["position"])
                if 1 <= pos < me.top_role.position:
                    try:
                        await _move_role(guild, role, pos, params["reason"])
                        landed = await _refetch_role(guild, role.id)
                        said.append(f"position {landed.position if landed else pos}")
                    except discord.HTTPException as e:
                        said.append(f"position NOT changed ({e})")
                else:
                    said.append(f"position {pos} refused (must be ≥ 1 and below my top role at {me.top_role.position})")

            perms = _privileged_perms(role)
            note = f"\n⚠ This role grants: {', '.join(perms)}." if perms else ""
            detail = (" — " + ", ".join(said)) if said else ""
            return [TextContent(
                type="text",
                text=f"Created role '{role.name}' (ID: {role.id}){detail}. Nobody holds it yet — hand it out with add_role.{note}"
            )]

        # ---- edit_role / delete_role: both name an existing role -----------------------------
        role_arg = str(arguments.get("role") or arguments.get("role_id") or "").strip()
        if not role_arg:
            return [TextContent(type="text", text="Error: which role? Pass `role` (name or ID).")]
        role = _resolve_role(guild, role_arg)
        if role is None:
            return [TextContent(
                type="text",
                text=f"Error: no role matching '{role_arg}' in this server. Call list_roles for the exact names and IDs."
            )]

        verb = "edit" if name == "edit_role" else "delete"
        blocked = _uneditable_reason(role, me)
        if blocked:
            return [TextContent(type="text", text=f"Cannot {verb} '{role.name}': {blocked}")]

        if name == "delete_role":
            holders, capped = await _role_holder_count(guild, role)
            perms = _privileged_perms(role)
            role_name, role_id = role.name, role.id
            try:
                await role.delete(reason=arguments.get("reason", "Role deleted via MCP"))
            except discord.Forbidden as e:
                return [TextContent(type="text", text=f"Discord refused to delete '{role_name}': {e}")]
            except discord.HTTPException as e:
                return [TextContent(type="text", text=f"Discord error while deleting '{role_name}': {e}")]

            if holders is None:
                lost = "I couldn't count how many members held it."
            else:
                lost = f"{holders}{'+' if capped else ''} member(s) lost it."
            note = f" It granted: {', '.join(perms)}." if perms else ""
            return [TextContent(
                type="text",
                text=f"Deleted role '{role_name}' (ID: {role_id}). {lost}{note} This cannot be undone."
            )]

        # ---- edit_role ----------------------------------------------------------------------
        edits, said = {}, []
        try:
            if arguments.get("name"):
                edits["name"] = arguments["name"]
                said.append(f"renamed to '{arguments['name']}'")
            if arguments.get("color") is not None:
                colour = _parse_colour(arguments["color"])
                edits["colour"] = colour
                said.append("colour cleared" if colour.value == 0 else f"colour {colour}")
            if arguments.get("hoist") is not None:
                edits["hoist"] = bool(arguments["hoist"])
                said.append("shown separately" if edits["hoist"] else "no longer shown separately")
            if arguments.get("mentionable") is not None:
                edits["mentionable"] = bool(arguments["mentionable"])
                said.append("mentionable" if edits["mentionable"] else "not mentionable")

            # Permissions: `permissions` replaces the whole set, grant/revoke nudge it. Both forms
            # end in one Permissions object, so the escalation check below sees the real diff.
            perms = None
            if arguments.get("permissions") is not None:
                perms = _permissions_from(arguments["permissions"])
            if arguments.get("grant_permissions") or arguments.get("revoke_permissions"):
                base = perms if perms is not None else discord.Permissions(role.permissions.value)
                for n in _parse_perm_names(arguments.get("grant_permissions") or [], "grant_permissions"):
                    setattr(base, n, True)
                for n in _parse_perm_names(arguments.get("revoke_permissions") or [], "revoke_permissions"):
                    setattr(base, n, False)
                perms = base
            if perms is not None:
                # Discord requires every permission you CHANGE (either way) to be one you hold
                # yourself, so the check is on the diff — not on the role's whole set, which may
                # legitimately contain powers I lack and am not touching.
                added = [n for n, on in perms if on and not getattr(role.permissions, n, False)]
                removed = [n for n, on in perms if not on and getattr(role.permissions, n, False)]
                if added or removed:
                    esc = _escalation_reason(discord.Permissions(**{n: True for n in added + removed}), me)
                    if esc:
                        return [TextContent(type="text", text=f"Cannot edit '{role.name}': {esc}")]
                    edits["permissions"] = perms
                    if added:
                        said.append("grants " + ", ".join(sorted(added)))
                    if removed:
                        said.append("no longer grants " + ", ".join(sorted(removed)))

            if arguments.get("clear_icon"):
                edits["display_icon"] = None
                said.append("icon removed")
            elif arguments.get("icon_emoji") or arguments.get("icon_file_path") or arguments.get("icon_url"):
                if not _role_icons_available(guild):
                    return [TextContent(
                        type="text",
                        text=f"Cannot set an icon on '{role.name}': {guild.name} doesn't have role icons "
                             "(Discord unlocks them at server boost level 2)."
                    )]
                edits["display_icon"], icon_said = await _role_icon_from(
                    guild, arguments.get("icon_emoji"), arguments.get("icon_file_path"), arguments.get("icon_url"))
                said.append(f"icon {icon_said}")

            # Position: absolute, or relative to another role. `above`/`below` are what a request
            # actually means ("выше Соседей") — an absolute number alone is easy to get backwards,
            # since Discord counts UP from @everyone at 0. The move goes through the bulk endpoint
            # (see _move_role) and is reported from a re-read, not from what we asked for.
            move_to, move_said = arguments.get("position"), None
            for key, delta in (("above", +1), ("below", -1)):
                if arguments.get(key):
                    target = _resolve_role(guild, arguments[key])
                    if target is None:
                        return [TextContent(
                            type="text",
                            text=f"Error: no role matching '{arguments[key]}' to place '{role.name}' {key}."
                        )]
                    move_to, move_said = target.position + delta, f"{key} '{target.name}'"
                    break
            if move_to is not None:
                move_to = int(move_to)
                if move_to < 1:
                    return [TextContent(
                        type="text",
                        text="Error: position 0 is @everyone — nothing can sit at or below it. Use 1 for the lowest real role."
                    )]
                if move_to >= me.top_role.position:
                    return [TextContent(
                        type="text",
                        text=f"Cannot move '{role.name}' to position {move_to}: my own top role "
                             f"'{me.top_role.name}' sits at {me.top_role.position}, and Discord only lets me "
                             "arrange roles strictly beneath it."
                    )]
        except (ValueError, RuntimeError) as e:
            return [TextContent(type="text", text=f"Error editing '{role.name}': {e}")]

        if not edits and move_to is None:
            return [TextContent(
                type="text",
                text="Error: nothing to change — pass name, color, hoist, mentionable, permissions/"
                     "grant_permissions/revoke_permissions, an icon (icon_emoji/icon_file_path/icon_url/"
                     "clear_icon) or a position (position/above/below)."
            )]

        reason = arguments.get("reason", "Role edited via MCP")
        try:
            if edits:
                await role.edit(reason=reason, **edits)
            if move_to is not None:
                await _move_role(guild, role, move_to, reason)
        except discord.Forbidden as e:
            return [TextContent(type="text", text=f"Discord refused to edit '{role.name}': {e}")]
        except discord.HTTPException as e:
            return [TextContent(type="text", text=f"Discord error while editing '{role.name}': {e}")]

        # Report from a re-read, never from the request: a move can land somewhere other than the
        # number asked for (Discord renumbers neighbours), and saying "moved above X" when it didn't
        # is worse than saying nothing.
        fresh = await _refetch_role(guild, role.id) or role
        if move_to is not None:
            said.append(f"{move_said} (now position {fresh.position})" if move_said
                        else f"position {fresh.position}")
        note = ""
        if "permissions" in edits and _privileged_perms(fresh):
            note = f"\n⚠ This role now grants: {', '.join(_privileged_perms(fresh))}."
        return [TextContent(
            type="text",
            text=f"Role '{role.name}' (ID: {role.id}): " + ", ".join(said) + ".\n"
                 f"Now: {_role_line(fresh, me)}{note}"
        )]

    # Custom Emoji Tools
    elif name == "list_emojis":
        server_id = arguments.get("server_id", DEFAULT_SERVER_ID)
        if not server_id:
            return [TextContent(
                type="text",
                text="Error: No server ID provided and no default server ID set. Set DEFAULT_SERVER_ID environment variable or provide server_id in the request."
            )]

        guild = await discord_client.fetch_guild(int(server_id))

        try:
            emojis = await guild.fetch_emojis()
            if not emojis:
                return [TextContent(type="text", text="No custom emojis on this server.")]

            static = sum(1 for e in emojis if not e.animated)
            animated = len(emojis) - static
            result = f"Custom emojis ({static} static, {animated} animated):\n"
            for emoji in sorted(emojis, key=lambda e: e.name.lower()):
                result += "- " + _emoji_line(emoji) + "\n"

            return [TextContent(
                type="text",
                text=result
            )]
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error listing emojis: {str(e)}"
            )]

    elif name == "create_emoji":
        server_id = arguments.get("server_id", DEFAULT_SERVER_ID)
        if not server_id:
            return [TextContent(
                type="text",
                text="Error: No server ID provided and no default server ID set. Set DEFAULT_SERVER_ID environment variable or provide server_id in the request."
            )]

        guild = await discord_client.fetch_guild(int(server_id))

        try:
            image = await _emoji_image_bytes(arguments.get("file_path"), arguments.get("url"))

            params = {
                "name": arguments["name"],
                "image": image,
                "reason": arguments.get("reason", "Emoji created via MCP"),
            }
            if arguments.get("roles"):
                params["roles"] = _find_roles(await guild.fetch_roles(), arguments["roles"])

            emoji = await guild.create_custom_emoji(**params)

            return [TextContent(
                type="text",
                text=f"Created emoji {_emoji_line(emoji)}"
            )]
        except discord.errors.Forbidden:
            return [TextContent(
                type="text",
                text="Bot doesn't have permission to manage emojis. It needs the 'Manage Expressions' permission on this server."
            )]
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error creating emoji: {str(e)}"
            )]

    elif name == "edit_emoji":
        server_id = arguments.get("server_id", DEFAULT_SERVER_ID)
        if not server_id:
            return [TextContent(
                type="text",
                text="Error: No server ID provided and no default server ID set. Set DEFAULT_SERVER_ID environment variable or provide server_id in the request."
            )]

        guild = await discord_client.fetch_guild(int(server_id))

        try:
            emoji = _find_emoji(await guild.fetch_emojis(), arguments["emoji"])
            if not emoji:
                return [TextContent(
                    type="text",
                    text=f"Error: No custom emoji matching '{arguments['emoji']}' on this server."
                )]

            params = {"reason": arguments.get("reason", "Emoji edited via MCP")}
            if "name" in arguments:
                params["name"] = arguments["name"]
            # An explicit empty list clears the restriction, so test for presence, not truthiness.
            if arguments.get("roles") is not None:
                params["roles"] = _find_roles(await guild.fetch_roles(), arguments["roles"])
            if len(params) == 1:
                return [TextContent(
                    type="text",
                    text="Error: nothing to change — pass name and/or roles."
                )]

            edited = await emoji.edit(**params)

            return [TextContent(
                type="text",
                text=f"Edited emoji {_emoji_line(edited)}"
            )]
        except discord.errors.Forbidden:
            return [TextContent(
                type="text",
                text="Bot doesn't have permission to manage emojis. It needs the 'Manage Expressions' permission on this server."
            )]
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error editing emoji: {str(e)}"
            )]

    elif name == "delete_emoji":
        server_id = arguments.get("server_id", DEFAULT_SERVER_ID)
        if not server_id:
            return [TextContent(
                type="text",
                text="Error: No server ID provided and no default server ID set. Set DEFAULT_SERVER_ID environment variable or provide server_id in the request."
            )]

        guild = await discord_client.fetch_guild(int(server_id))

        try:
            emoji = _find_emoji(await guild.fetch_emojis(), arguments["emoji"])
            if not emoji:
                return [TextContent(
                    type="text",
                    text=f"Error: No custom emoji matching '{arguments['emoji']}' on this server."
                )]

            emoji_name, emoji_id = emoji.name, emoji.id
            await emoji.delete(reason=arguments.get("reason", "Emoji deleted via MCP"))

            return [TextContent(
                type="text",
                text=f"Deleted emoji {emoji_name} (ID: {emoji_id})"
            )]
        except discord.errors.Forbidden:
            return [TextContent(
                type="text",
                text="Bot doesn't have permission to manage emojis. It needs the 'Manage Expressions' permission on this server."
            )]
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error deleting emoji: {str(e)}"
            )]

    # Sticker Tools
    elif name == "list_stickers":
        server_id = arguments.get("server_id", DEFAULT_SERVER_ID)
        if not server_id:
            return [TextContent(
                type="text",
                text="Error: No server ID provided and no default server ID set. Set DEFAULT_SERVER_ID environment variable or provide server_id in the request."
            )]

        guild = await discord_client.fetch_guild(int(server_id))

        try:
            stickers = await guild.fetch_stickers()
            if not stickers:
                return [TextContent(
                    type="text",
                    text=f"No custom stickers on this server (0 of {guild.sticker_limit} slots used)."
                )]

            result = f"Custom stickers ({len(stickers)} of {guild.sticker_limit} slots used):\n"
            for sticker in sorted(stickers, key=lambda s: s.name.lower()):
                result += "- " + _sticker_line(sticker) + "\n"

            return [TextContent(
                type="text",
                text=result
            )]
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error listing stickers: {str(e)}"
            )]

    elif name == "create_sticker":
        server_id = arguments.get("server_id", DEFAULT_SERVER_ID)
        if not server_id:
            return [TextContent(
                type="text",
                text="Error: No server ID provided and no default server ID set. Set DEFAULT_SERVER_ID environment variable or provide server_id in the request."
            )]

        guild = await discord_client.fetch_guild(int(server_id))

        try:
            import io
            raw = await _read_image_source(arguments.get("file_path"), arguments.get("url"))
            image = _fit_sticker_image(raw)

            sticker = await guild.create_sticker(
                name=arguments["name"],
                description=arguments.get("description", ""),
                emoji=arguments["emoji"],
                file=discord.File(io.BytesIO(image), filename="sticker.png"),
                reason=arguments.get("reason", "Sticker created via MCP"),
            )

            return [TextContent(
                type="text",
                text=f"Created sticker {_sticker_line(sticker)}. Send it with send_sticker."
            )]
        except discord.errors.Forbidden:
            return [TextContent(
                type="text",
                text="Bot doesn't have permission to manage stickers. It needs the 'Manage Expressions' permission on this server."
            )]
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error creating sticker: {str(e)}"
            )]

    elif name == "edit_sticker":
        server_id = arguments.get("server_id", DEFAULT_SERVER_ID)
        if not server_id:
            return [TextContent(
                type="text",
                text="Error: No server ID provided and no default server ID set. Set DEFAULT_SERVER_ID environment variable or provide server_id in the request."
            )]

        guild = await discord_client.fetch_guild(int(server_id))

        try:
            sticker = _find_sticker(await guild.fetch_stickers(), arguments["sticker"])
            if not sticker:
                return [TextContent(
                    type="text",
                    text=f"Error: No sticker matching '{arguments['sticker']}' on this server."
                )]

            params = {"reason": arguments.get("reason", "Sticker edited via MCP")}
            for field in ("name", "description", "emoji"):
                if field in arguments:
                    params[field] = arguments[field]
            if len(params) == 1:
                return [TextContent(
                    type="text",
                    text="Error: nothing to change — pass name, description and/or emoji."
                )]

            edited = await sticker.edit(**params)

            return [TextContent(
                type="text",
                text=f"Edited sticker {_sticker_line(edited)}"
            )]
        except discord.errors.Forbidden:
            return [TextContent(
                type="text",
                text="Bot doesn't have permission to manage stickers. It needs the 'Manage Expressions' permission on this server."
            )]
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error editing sticker: {str(e)}"
            )]

    elif name == "delete_sticker":
        server_id = arguments.get("server_id", DEFAULT_SERVER_ID)
        if not server_id:
            return [TextContent(
                type="text",
                text="Error: No server ID provided and no default server ID set. Set DEFAULT_SERVER_ID environment variable or provide server_id in the request."
            )]

        guild = await discord_client.fetch_guild(int(server_id))

        try:
            sticker = _find_sticker(await guild.fetch_stickers(), arguments["sticker"])
            if not sticker:
                return [TextContent(
                    type="text",
                    text=f"Error: No sticker matching '{arguments['sticker']}' on this server."
                )]

            sticker_name, sticker_id = sticker.name, sticker.id
            await sticker.delete(reason=arguments.get("reason", "Sticker deleted via MCP"))

            return [TextContent(
                type="text",
                text=f"Deleted sticker {sticker_name} (ID: {sticker_id})"
            )]
        except discord.errors.Forbidden:
            return [TextContent(
                type="text",
                text="Bot doesn't have permission to manage stickers. It needs the 'Manage Expressions' permission on this server."
            )]
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error deleting sticker: {str(e)}"
            )]

    elif name == "send_sticker":
        channel = await discord_client.fetch_channel(int(arguments["channel_id"]))
        ref = str(arguments["sticker"]).strip()

        try:
            if ref.isdigit():
                # A bare ID may be any sticker the bot can use, not only this guild's.
                sticker = await discord_client.fetch_sticker(int(ref))
            else:
                server_id = arguments.get("server_id", DEFAULT_SERVER_ID)
                if not server_id:
                    return [TextContent(
                        type="text",
                        text="Error: sticker given by name needs a server to resolve against — set DEFAULT_SERVER_ID or pass server_id."
                    )]
                guild = await discord_client.fetch_guild(int(server_id))
                sticker = _find_sticker(await guild.fetch_stickers(), ref)
                if not sticker:
                    return [TextContent(
                        type="text",
                        text=f"Error: No sticker matching '{ref}' on this server."
                    )]

            message = await channel.send(content=arguments.get("content") or None, stickers=[sticker])
            _stop_typing(int(arguments["channel_id"]))   # a sticker is a reply too → stop the typing loop
            return [TextContent(
                type="text",
                text=f"Sticker {sticker.name} sent. Message ID: {message.id}"
            )]
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error sending sticker: {str(e)}"
            )]

    # User Management Tools
    elif name == "kick_user":
        server_id = arguments.get("server_id", DEFAULT_SERVER_ID)
        if not server_id:
            return [TextContent(
                type="text",
                text="Error: No server ID provided and no default server ID set. Set DEFAULT_SERVER_ID environment variable or provide server_id in the request."
            )]
            
        guild = await discord_client.fetch_guild(int(server_id))
        
        try:
            # Get the member from the guild
            member = await guild.fetch_member(int(arguments["user_id"]))
            
            # Kick the member
            reason = arguments.get("reason", "Kicked via MCP")
            await member.kick(reason=reason)
            
            return [TextContent(
                type="text",
                text=f"Successfully kicked user {member.name}#{member.discriminator} (ID: {member.id}) from the server."
            )]
        except discord.errors.NotFound:
            return [TextContent(
                type="text",
                text=f"User with ID {arguments['user_id']} not found in the server."
            )]
        except discord.errors.Forbidden:
            return [TextContent(
                type="text",
                text="Bot doesn't have permission to kick this user. Make sure the bot's role is higher than the user's role."
            )]
    
    elif name == "ban_user":
        server_id = arguments.get("server_id", DEFAULT_SERVER_ID)
        if not server_id:
            return [TextContent(
                type="text",
                text="Error: No server ID provided and no default server ID set. Set DEFAULT_SERVER_ID environment variable or provide server_id in the request."
            )]
            
        guild = await discord_client.fetch_guild(int(server_id))
        
        try:
            # Get optional parameters
            reason = arguments.get("reason", "Banned via MCP")
            delete_message_days = min(int(arguments.get("delete_message_days", 0)), 7)
            
            # Ban the user
            await guild.ban(
                discord.Object(id=int(arguments["user_id"])),
                reason=reason,
                delete_message_days=delete_message_days
            )
            
            return [TextContent(
                type="text",
                text=f"Successfully banned user with ID {arguments['user_id']} from the server. Deleted messages from the past {delete_message_days} days."
            )]
        except discord.errors.NotFound:
            return [TextContent(
                type="text",
                text=f"User with ID {arguments['user_id']} not found."
            )]
        except discord.errors.Forbidden:
            return [TextContent(
                type="text",
                text="Bot doesn't have permission to ban this user. Make sure the bot's role is higher than the user's role."
            )]

    raise ValueError(f"Unknown tool: {name}")

async def main():
    # Start Discord bot in the background
    asyncio.create_task(bot.start(DISCORD_TOKEN))
    
    # Run MCP server
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
