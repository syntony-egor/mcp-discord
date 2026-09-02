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


async def _emoji_image_bytes(file_path=None, url=None) -> bytes:
    """Local path or image URL -> bytes that fit Discord's emoji size cap."""
    if file_path:
        from pathlib import Path as _P
        raw = _P(file_path).read_bytes()
    elif url:
        import aiohttp
        async with aiohttp.ClientSession() as sess:
            async with sess.get(url) as r:
                if r.status != 200:
                    raise RuntimeError(f"image download failed ({r.status}): {url}")
                raw = await r.read()
    else:
        raise ValueError("either file_path or url is required")

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
            description="Add a role to a user",
            inputSchema={
                "type": "object",
                "properties": {
                    "server_id": {
                        "type": "string",
                        "description": "Discord server ID. If not provided, the default server ID will be used."
                    },
                    "user_id": {
                        "type": "string",
                        "description": "User to add role to"
                    },
                    "role_id": {
                        "type": "string",
                        "description": "Role ID to add"
                    }
                },
                "required": ["user_id", "role_id"]
            }
        ),
        Tool(
            name="remove_role",
            description="Remove a role from a user",
            inputSchema={
                "type": "object",
                "properties": {
                    "server_id": {
                        "type": "string",
                        "description": "Discord server ID. If not provided, the default server ID will be used."
                    },
                    "user_id": {
                        "type": "string",
                        "description": "User to remove role from"
                    },
                    "role_id": {
                        "type": "string",
                        "description": "Role ID to remove"
                    }
                },
                "required": ["user_id", "role_id"]
            }
        ),

        # Channel Management Tools
        Tool(
            name="create_text_channel",
            description="Create a new text channel",
            inputSchema={
                "type": "object",
                "properties": {
                    "server_id": {
                        "type": "string",
                        "description": "Discord server ID. If not provided, the default server ID will be used."
                    },
                    "name": {
                        "type": "string",
                        "description": "Channel name"
                    },
                    "category_id": {
                        "type": "string",
                        "description": "Optional category ID to place channel in"
                    },
                    "topic": {
                        "type": "string",
                        "description": "Optional channel topic"
                    }
                },
                "required": ["name"]
            }
        ),
        Tool(
            name="delete_channel",
            description="Delete a channel",
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
            description="Rename a channel (e.g. change the leading emoji of a diary). Requires Manage Channels.",
            inputSchema={
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "string",
                        "description": "ID of the channel to rename"
                    },
                    "name": {
                        "type": "string",
                        "description": "New channel name (full name, including any leading emoji)"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Optional audit-log reason"
                    }
                },
                "required": ["channel_id", "name"]
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
            description="Start a CONTINUOUS 'Bot is typing…' indicator in a channel — re-triggered every ~8s so it stays visible the whole time you think. Auto-stops when you send a message there (or call stop_typing). Call right when you begin composing a reply.",
            inputSchema={
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "string",
                        "description": "Channel to show the typing indicator in"
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
            description="Get information about a Discord user, including their avatar/banner CDN URLs (download the returned Avatar URL with download_attachment).",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "Discord user ID"
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
            description="Create a new role in the server",
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
                        "description": "Color of the role in hex format (e.g., '#FF0000' for red)"
                    },
                    "hoist": {
                        "type": "boolean",
                        "description": "Whether the role should be displayed separately in the member list"
                    },
                    "mentionable": {
                        "type": "boolean",
                        "description": "Whether the role can be mentioned by anyone"
                    },
                    "permissions": {
                        "type": "string",
                        "description": "Permissions as an integer (optional)"
                    }
                },
                "required": ["name"]
            }
        ),
        Tool(
            name="delete_role",
            description="Delete a role from the server",
            inputSchema={
                "type": "object",
                "properties": {
                    "server_id": {
                        "type": "string",
                        "description": "Discord server ID. If not provided, the default server ID will be used."
                    },
                    "role_id": {
                        "type": "string",
                        "description": "ID of the role to delete"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for deleting the role"
                    }
                },
                "required": ["role_id"]
            }
        ),
        Tool(
            name="list_roles",
            description="List all roles in the server",
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
        _stop_typing(int(arguments["channel_id"]))   # reply sent → stop the typing loop
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
        return [TextContent(
            type="text",
            text=f"File sent successfully. Message ID: {message.id}"
        )]

    elif name == "send_voice_message":
        message_id = await _send_voice_message(int(arguments["channel_id"]), arguments["file_path"])
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
                 f"Banner URL: {banner_url or '(none)'}"
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
    elif name == "add_role":
        server_id = arguments.get("server_id", DEFAULT_SERVER_ID)
        if not server_id:
            return [TextContent(
                type="text",
                text="Error: No server ID provided and no default server ID set. Set DEFAULT_SERVER_ID environment variable or provide server_id in the request."
            )]
            
        guild = await discord_client.fetch_guild(int(server_id))
        member = await guild.fetch_member(int(arguments["user_id"]))
        role = guild.get_role(int(arguments["role_id"]))
        
        await member.add_roles(role, reason="Role added via MCP")
        return [TextContent(
            type="text",
            text=f"Added role {role.name} to user {member.name}"
        )]

    elif name == "remove_role":
        server_id = arguments.get("server_id", DEFAULT_SERVER_ID)
        if not server_id:
            return [TextContent(
                type="text",
                text="Error: No server ID provided and no default server ID set. Set DEFAULT_SERVER_ID environment variable or provide server_id in the request."
            )]
            
        guild = await discord_client.fetch_guild(int(server_id))
        member = await guild.fetch_member(int(arguments["user_id"]))
        role = guild.get_role(int(arguments["role_id"]))
        
        await member.remove_roles(role, reason="Role removed via MCP")
        return [TextContent(
            type="text",
            text=f"Removed role {role.name} from user {member.name}"
        )]

    # Channel Management Tools
    elif name == "create_text_channel":
        server_id = arguments.get("server_id", DEFAULT_SERVER_ID)
        if not server_id:
            return [TextContent(
                type="text",
                text="Error: No server ID provided and no default server ID set. Set DEFAULT_SERVER_ID environment variable or provide server_id in the request."
            )]
            
        guild = await discord_client.fetch_guild(int(server_id))
        category = None
        if "category_id" in arguments:
            try:
                # Properly fetch the category instead of using get_channel which only checks cache
                category = await discord_client.fetch_channel(int(arguments["category_id"]))
                if not isinstance(category, discord.CategoryChannel):
                    logger.warning(f"Channel {arguments['category_id']} is not a category channel")
                    category = None
            except Exception as e:
                logger.error(f"Error fetching category: {str(e)}")
                category = None
        
        channel = await guild.create_text_channel(
            name=arguments["name"],
            category=category,
            topic=arguments.get("topic"),
            reason="Channel created via MCP"
        )
        
        return [TextContent(
            type="text",
            text=f"Created text channel #{channel.name} (ID: {channel.id})"
        )]

    elif name == "delete_channel":
        channel = await discord_client.fetch_channel(int(arguments["channel_id"]))
        await channel.delete(reason=arguments.get("reason", "Channel deleted via MCP"))
        return [TextContent(
            type="text",
            text=f"Deleted channel successfully"
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
        channel = await discord_client.fetch_channel(int(arguments["channel_id"]))
        new_name = arguments["name"]
        try:
            # Discord rate-limits channel renames to 2 / 10 min. On 429 discord.py does NOT
            # raise — it silently sleeps retry_after (often hundreds of seconds) and retries,
            # hanging this tool call. Bound the wait: a normal rename is < 1s; not done in
            # time → we hit the limit, so cancel and tell the caller to try later.
            await asyncio.wait_for(
                channel.edit(name=new_name, reason=arguments.get("reason", "diary-emoji")),
                timeout=5)
        except (asyncio.TimeoutError, discord.RateLimited):
            return [TextContent(
                type="text",
                text="Rename rate-limited (Discord allows 2 renames / 10 min). Try later."
            )]
        except discord.Forbidden:
            return [TextContent(
                type="text",
                text="Error: Bot doesn't have permission to rename this channel"
            )]
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error renaming channel: {str(e)}"
            )]
        return [TextContent(
            type="text",
            text=f"Channel renamed to {new_name}"
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
                reason="Category created via MCP"
            )
            
            # Create a default text channel in the category to make it more visible
            text_channel = await guild.create_text_channel(
                name=f"{arguments['name']}-general",
                category=category,
                reason="Default channel for new category"
            )
            
            # Generate appropriate success message
            if restricted_role_id and not everyone_can_view:
                msg = f"Created restricted category {category.name} (ID: {category.id}) with default channel #{text_channel.name}. Only specified roles can view it."
            elif not everyone_can_view:
                msg = f"Created hidden category {category.name} (ID: {category.id}) with default channel #{text_channel.name}. @everyone cannot view it."
            else:
                msg = f"Created category {category.name} (ID: {category.id}) with default channel #{text_channel.name}"
                
            return [TextContent(
                type="text",
                text=msg
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
    elif name == "create_role":
        server_id = arguments.get("server_id", DEFAULT_SERVER_ID)
        if not server_id:
            return [TextContent(
                type="text",
                text="Error: No server ID provided and no default server ID set. Set DEFAULT_SERVER_ID environment variable or provide server_id in the request."
            )]
            
        guild = await discord_client.fetch_guild(int(server_id))
        
        try:
            # Set up role creation parameters
            params = {
                "name": arguments["name"],
                "reason": "Role created via MCP"
            }
            
            # Handle optional parameters
            if "color" in arguments:
                color_str = arguments["color"].lstrip('#')
                color_int = int(color_str, 16)
                params["colour"] = discord.Colour(color_int)
                
            if "hoist" in arguments:
                params["hoist"] = arguments["hoist"]
                
            if "mentionable" in arguments:
                params["mentionable"] = arguments["mentionable"]
                
            if "permissions" in arguments:
                permissions_int = int(arguments["permissions"])
                params["permissions"] = discord.Permissions(permissions=permissions_int)
            
            # Create the role
            role = await guild.create_role(**params)
            
            return [TextContent(
                type="text",
                text=f"Created role {role.name} (ID: {role.id})"
            )]
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error creating role: {str(e)}"
            )]
    
    elif name == "delete_role":
        server_id = arguments.get("server_id", DEFAULT_SERVER_ID)
        if not server_id:
            return [TextContent(
                type="text",
                text="Error: No server ID provided and no default server ID set. Set DEFAULT_SERVER_ID environment variable or provide server_id in the request."
            )]
            
        guild = await discord_client.fetch_guild(int(server_id))
        role_id = int(arguments["role_id"])
        
        try:
            # Find the role
            role = guild.get_role(role_id)
            if not role:
                roles = await guild.fetch_roles()
                for r in roles:
                    if r.id == role_id:
                        role = r
                        break
            
            if not role:
                return [TextContent(
                    type="text",
                    text=f"Error: Role with ID {role_id} not found in the server."
                )]
            
            # Delete the role
            role_name = role.name
            await role.delete(reason=arguments.get("reason", "Role deleted via MCP"))
            
            return [TextContent(
                type="text",
                text=f"Deleted role {role_name} (ID: {role_id})"
            )]
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error deleting role: {str(e)}"
            )]
    
    elif name == "list_roles":
        server_id = arguments.get("server_id", DEFAULT_SERVER_ID)
        if not server_id:
            return [TextContent(
                type="text",
                text="Error: No server ID provided and no default server ID set. Set DEFAULT_SERVER_ID environment variable or provide server_id in the request."
            )]
            
        guild = await discord_client.fetch_guild(int(server_id))
        
        try:
            # Fetch all roles from the guild
            roles = await guild.fetch_roles()
            
            # Format role information
            role_info = []
            for role in roles:
                role_info.append({
                    "id": str(role.id),
                    "name": role.name,
                    "color": str(role.color),
                    "position": role.position,
                    "hoisted": role.hoist,
                    "mentionable": role.mentionable
                })
            
            # Sort roles by position (higher positions are higher in the hierarchy)
            role_info.sort(key=lambda r: r["position"], reverse=True)
            
            # Format the output
            result = "Server Roles:\n"
            for role in role_info:
                result += f"- {role['name']} (ID: {role['id']}, Position: {role['position']})\n"
            
            return [TextContent(
                type="text",
                text=result
            )]
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error listing roles: {str(e)}"
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
