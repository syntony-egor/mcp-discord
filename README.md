# mcp-discord

A [Model Context Protocol](https://modelcontextprotocol.io) (MCP) server that gives AI agents — Claude Code, Claude Desktop, Goose, or any MCP-capable client — full read/write access to a Discord server: send and read messages, react, manage channels and roles, upload and download files, and show a live "typing…" indicator.

This is a **fork of [netixc/mcp-discord](https://github.com/netixc/mcp-discord)** (originally by Jawad Bly), with extra tools added for richer agent behavior:

- `start_typing` / `stop_typing` — continuous "Bot is typing…" indicator
- `set_presence` — set the bot's online/offline dot (`online` / `idle` / `dnd` / `invisible`, optional status text)
- `send_file` — upload a local file to a channel
- `send_voice_message` — send a NATIVE voice message (waveform bubble; any audio in, ffmpeg required)
- `download_attachment` — save a Discord attachment to a local path
- `move_channel` — reposition a channel / move it between categories
- `edit_channel` — rename a channel (e.g. change a diary's leading emoji)
- `edit_message` — rewrite one of the bot's own earlier messages (e.g. keep a channel-side digest of a thread current)
- `list_emojis` / `create_emoji` / `edit_emoji` / `delete_emoji` — manage the server's custom emojis (upload from a file or URL; oversized images are auto-resized)
- `list_stickers` / `create_sticker` / `edit_sticker` / `delete_sticker` / `send_sticker` — manage and post the server's custom stickers (auto-fitted to Discord's exact 320×320 / 512 KB)

Set `MCP_DISCORD_CONNECT_INVISIBLE=1` to make the bot **connect `invisible`**, so merely running this server does not
light the bot up — a client then lights it with `set_presence("online")` and darkens it with `set_presence("invisible")`.
Without the flag the bot keeps discord.py's default (online), so this never silently darkens an existing consumer.

These additions are what the **[muraveynik](https://github.com/syntony-egor/muraveynik)** responder relies on (typing presence while the agent composes, exchanging images and documents) — see [Used by](#used-by).

## Requirements

- **Python ≥ 3.10**
- **[uv](https://docs.astral.sh/uv/)** (package/run manager) — install with `curl -LsSf https://astral.sh/uv/install.sh | sh`
- A Discord **bot token** with the bot added to your server (see [Get a bot token](#get-a-bot-token))

Runtime dependencies (resolved by uv from `uv.lock` / `pyproject.toml`): `discord.py>=2.3.0`, `mcp>=0.1.0`, `pillow>=10.0.0` (downscales oversized `create_emoji` images).

- **ffmpeg** — required only by `send_voice_message` (transcode + waveform).

> **Python 3.13+ only:** `discord.py`'s voice support imports `audioop`, removed from the stdlib in 3.13. If you are on 3.13+, add the shim to the project: `uv add audioop-lts`. (Not needed on 3.10–3.12.)

## Install

```bash
git clone git@github.com:syntony-egor/mcp-discord.git
cd mcp-discord
uv sync          # creates .venv and installs locked dependencies
```

Smoke-test that the package imports. With no `DISCORD_TOKEN` set, the server **raises at import time** — that is the expected, correct failure (it proves the dependency tree resolves):

```bash
uv run mcp-discord 2>&1 | grep -m1 DISCORD_TOKEN || true
# -> ValueError: DISCORD_TOKEN environment variable is required   (expected without a token)
```

(`grep -m1 … || true` keeps the check robust against SIGPIPE on the piped traceback.)

The server speaks MCP over **stdio**. You normally do not run it by hand — an MCP client launches it for you (see [Wire it into an MCP client](#wire-it-into-an-mcp-client)).

## Configuration

Configured entirely through environment variables:

| Variable | Required | Purpose |
|---|---|---|
| `DISCORD_TOKEN` | **yes** | Bot token the server logs in with. Read at import; the server refuses to start without it. |
| `DEFAULT_SERVER_ID` | no (recommended) | Guild (server) ID used by server/role/member tools when `server_id` is omitted. Without it, those tools error unless every call passes `server_id`. |

### Get a bot token

1. <https://discord.com/developers/applications> → **New Application**, name it.
2. **Bot** tab → **Reset Token** → copy it (treat it like a password — anyone with it controls the bot; never commit it).
3. **Bot** tab → **Privileged Gateway Intents** → enable both (the server initializes the gateway with `message_content` and `members` intents, so it will **fail to connect** if these are off):
   - **MESSAGE CONTENT INTENT** — required to read message text via `read_messages`.
   - **SERVER MEMBERS INTENT** — required for `list_members` and member lookups.
4. **OAuth2 → URL Generator** → scope **bot**, then tick the permissions for the tools you'll use:
   - Read/send: **View Channels**, **Read Message History**, **Send Messages**, **Add Reactions**, **Attach Files**
   - Moderation/admin (optional): **Manage Channels**, **Manage Roles**, **Kick Members**, **Ban Members**, **Moderate Members**, **Manage Messages**
5. Open the generated URL and add the bot to your server (you must be a server admin).

### Find the server ID

In Discord: **User Settings → Advanced → Developer Mode** (on), then right-click the **server icon → Copy Server ID**. Use it as `DEFAULT_SERVER_ID`.

## Wire it into an MCP client

Add an entry to your client's MCP config, pointing `uv --directory` at this checkout. Works for Claude Code (`.mcp.json` in a project), Claude Desktop (`claude_desktop_config.json`), and any other MCP client.

```json
{
  "mcpServers": {
    "discord": {
      "command": "/home/egor/.local/bin/uv",
      "args": [
        "--directory",
        "/home/egor/mcp-discord",
        "run",
        "mcp-discord"
      ],
      "env": {
        "DISCORD_TOKEN": "<your bot token>",
        "DEFAULT_SERVER_ID": "<your guild id>"
      }
    }
  }
}
```

Notes:
- `command` should be the absolute path to your `uv` binary (`which uv`). A bare `"uv"` works only if `uv` is on the client's `PATH`.
- The `--directory` path **must** point at this repo checkout — keep it in sync if you move the clone.
- Claude Desktop on Windows/macOS uses the same shape with adjusted paths (`C:\\path\\to\\mcp-discord`, etc.).

After the client restarts/reconnects, the tools appear namespaced as `mcp__discord__send_message`, `mcp__discord__read_messages`, and so on.

## Tools

All IDs (channel, message, user, role, server) are passed as **strings**.

### Messaging
- `send_message` — send a message to a channel (`channel_id`, `content`). Also stops any active typing indicator in that channel.
- `edit_message` *(fork addition)* — replace the content of a message **the bot itself posted** (`channel_id`, `message_id`, `content`); Discord forbids editing anyone else's message, so a `Forbidden` here means "not our message".
- `read_messages` — read recent history (`channel_id`, `limit` ≤ 100); returns author, timestamp, content, reactions, and attachment URLs.

### Reactions
- `add_reaction` — add one reaction (`channel_id`, `message_id`, `emoji`).
- `add_multiple_reactions` — add several reactions at once (`emojis` array).
- `remove_reaction` — remove the bot's own reaction.

### Typing *(fork addition)*
- `start_typing` — start a **continuous** "typing…" indicator (auto-refreshed ~every 8 s, safety-capped at 5 min). Call it when the agent begins composing; it auto-stops on the next `send_message` or reaction in that channel.
- `stop_typing` — stop it manually (rarely needed).

### Presence *(fork addition)*
- `set_presence` — set the bot's account-wide presence dot: `status` = `online` / `idle` / `dnd` / `invisible`, plus optional `activity_text` for a custom status line. With `MCP_DISCORD_CONNECT_INVISIBLE=1` the bot **connects `invisible`** and stays dark until a client sets it `online`; use `invisible` to make it appear offline again while staying connected. The last-set presence is re-asserted across gateway reconnects, so it survives a re-IDENTIFY.

### Files & attachments *(fork addition)*
- `send_file` — upload a local file (`channel_id`, `file_path`, optional `content`).
- `send_voice_message` — native voice message (`channel_id`, `file_path`). Uses the cloud-attachment
  upload flow + `flags=8192`, so it renders as a voice bubble rather than an audio attachment; the
  input is transcoded to ogg/opus and the waveform measured with **ffmpeg** (required). Voice
  messages carry no text — send words separately.
- `download_attachment` — download an attachment URL (from `read_messages`) to a local `output_path`.

### Server & user info
- `get_server_info` — guild metadata + categories and text/voice channels with IDs.
- `list_members` — members with nicks, join dates, and role IDs (`limit` ≤ 1000).
- `get_user_info` — name, discriminator, bot flag, creation date, avatar/banner URLs for a `user_id`, plus their **server membership** (nickname, join date, roles held) when they are a member — the cheap way to see someone's current roles before changing them.

### Channels
- `create_text_channel`, `delete_channel`
- `create_category` (with optional restricted-role visibility)
- `create_thread` (from a message or standalone)
- `set_channel_permissions` (per-role view/send/read-history, optional @everyone)
- `move_channel` *(fork addition)* — reposition / move between categories, optionally syncing category permissions.
- `edit_channel` *(fork addition)* — rename a channel (full new name); bounded 5s wait so a rename rate-limit (2 / 10 min) returns cleanly instead of hanging.

### Roles
- `list_roles` — roles highest first, each marked **✅ assignable / ✋ not** (with the reason: integration-managed, at/above the bot's top role, missing *Manage Roles*) and flagged **⚠** when it grants staff permissions. Check this before assigning. To see *who* holds a role, use `list_members` with its `role` filter.
- `add_role`, `remove_role` — give/take a role on a member. `role` accepts the **name** (case-insensitive, `@` optional) or the ID, so no ID lookup is needed. Both refuse with the real cause instead of a bare 403, report a no-op harmlessly when the member already has (or already lacks) the role, take an audit-log `reason`, and warn when the role just granted staff powers.
- `create_role`, `delete_role`

### Emojis *(fork addition)*
- `list_emojis` — the server's custom emojis; each line gives the exact string to type in a message or pass to `add_reaction` (`<:name:id>`, `<a:name:id>` when animated) plus any role restriction.
- `create_emoji` — upload a new emoji from a local `file_path` **or** an image `url` (e.g. a freshly generated picture, or a Discord attachment URL). Images over Discord's **256 KB** cap are downscaled to 128/96/64 px until they fit (animated GIFs keep their frames), so a 3 MB render uploads without a manual resize. Optional `roles` restricts who may use it. Returns the `<:name:id>` string.
- `edit_emoji` — rename (`name`) and/or change the `roles` restriction; an empty `roles` array clears it.
- `delete_emoji` — remove an emoji. Irreversible.

> `edit_emoji` / `delete_emoji` take the emoji as an ID, a bare name, `:name:` or `<:name:id>`.
> All four need the bot to have **Manage Expressions** on the server.

### Stickers *(fork addition)*
- `list_stickers` — the server's stickers with name, ID, format, emoji tag and description, plus **slots used** (5 / 15 / 30 / 60 by boost level — far scarcer than emoji).
- `create_sticker` — upload from a local `file_path` or an image `url`. Discord demands **exactly 320×320** and **512 KB**, so the image is padded to square (letterboxed, never squashed), resized and — for photo-like art — palette-compressed until it fits; animated GIFs become APNG. Needs `name` (2-30) and `emoji`, one unicode emoji that tags the expression (Discord stores it by its unicode *name*, e.g. `😺` → `SMILING_CAT_FACE_WITH_OPEN_MOUTH`); `description` is optional.
- `edit_sticker` — change `name`, `description` and/or `emoji` tag.
- `delete_sticker` — remove a sticker. Irreversible.
- `send_sticker` — post a sticker to a channel (by ID, or by name on this server), with optional `content` in the same message. Stickers cannot be typed inline the way emojis can, so `send_message` will not do it.

> Stickers are **not** big emojis: different endpoint, different limits, and they ride the message rather than sitting in its text.

### Moderation / admin
- `moderate_message` — delete a message, optionally timeout the author.
- `kick_user`, `ban_user` (with optional message-deletion window).

> Server/role/member/category tools accept `server_id`; if omitted they fall back to `DEFAULT_SERVER_ID`, and error if neither is set.

## Used by

This fork is a dependency of **muraveynik** (the «Садик» Discord assistant). Its responder runs from an interactive Claude Code session and posts/reacts as the "Клод" bot through this server, specifically depending on the fork-added tools:
- `set_presence` — the responder (with `MCP_DISCORD_CONNECT_INVISIBLE=1` set in its launcher) lights the bot `online` on start and `invisible` on stop, so the online dot reflects "the responder is live";
- `start_typing` — the orchestrator shows "typing…" the instant a human ping arrives, so it's visible while the agent composes;
- `send_file` / `download_attachment` — exchange images and documents.

The upstream `netixc/mcp-discord` lacks these, so muraveynik pins this fork.

## Troubleshooting

- **`ValueError: DISCORD_TOKEN environment variable is required`** — the token isn't in the launched environment. Set `env.DISCORD_TOKEN` in the MCP client config (the client's own shell env is not inherited unless the client passes it).
- **`read_messages` returns empty / no content** — **MESSAGE CONTENT INTENT** is off in the Developer Portal, or the bot lacks **Read Message History** in that channel.
- **`list_members` errors or returns few members** — **SERVER MEMBERS INTENT** is off.
- **"No server ID provided and no default server ID set"** — set `DEFAULT_SERVER_ID` or pass `server_id` in the call.
- **`create_sticker` fails with "Maximum number of stickers reached"** — sticker slots are much scarcer than emoji ones (5 unboosted, 30 at boost level 2); `list_stickers` shows the count.
- **`create_emoji` fails with "Maximum number of emojis reached"** — the server is out of emoji slots (50 at boost level 0, more per boost level); delete one with `delete_emoji` first.
- **Forbidden / permission errors on channel/role/moderation tools** — the bot's OAuth permissions or role hierarchy are insufficient; the bot's role must sit above any role/user it manages.
- **Python 3.13+ `audioop` import error** — run `uv add audioop-lts` in this project (do not use `uv pip install` — that targets an ad-hoc env, not the `uv sync`-managed `.venv`).
- **Client can't launch the server** — confirm `command` is an absolute path to `uv` and `--directory` points at this checkout; test manually with `DISCORD_TOKEN=... uv --directory /path/to/mcp-discord run mcp-discord`.

## Credits & License

Fork of [netixc/mcp-discord](https://github.com/netixc/mcp-discord), originally created by **Jawad Bly**. Licensed under the **MIT License** — see [`LICENSE`](LICENSE) (© 2024 Jawad Bly). Modifications in this fork are released under the same license.
