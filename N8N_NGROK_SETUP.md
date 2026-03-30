# n8n Cloud + ngrok / Cloudflare: Fix SSL, 100s tunnel limit, and 60s execution timeout

You’re using **n8n Cloud** and your FastAPI is behind **ngrok** or **Cloudflare Tunnel**.

- **ngrok**: SSL/TLS errors (“decryption failed or bad record mac”) often come from long-lived connections or ngrok’s interstitial.
- **Cloudflare Tunnel**: Requests must complete within **~100 seconds** or Cloudflare disconnects.
- **n8n “Task execution timed out after 60 seconds”**: n8n aborts the workflow after 60 seconds by default. The pipeline can run **up to an hour or more** for long videos, so either increase the timeout or use the **callback** flow (no long-running execution).

---

## Option A: Callback flow (recommended – no n8n timeout)

The pipeline can take **up to an hour or more**. With the **callback** flow, n8n never runs longer than a few seconds: it submits the job and responds immediately; when the pipeline finishes, FastAPI **POSTs the result** to a second n8n webhook. No polling, no execution timeout.

1. **Import two workflows**
   - **`workflow-callback-trigger.json`** – receives the initial request (video URL), submits the job with a `callback_url`, responds with `job_id`.
   - **`workflow-callback-receive.json`** – webhook that FastAPI calls when the job is done; it receives the PDF (or error) and responds 200. You can add nodes here to save the PDF (e.g. Google Drive) or send an email.

2. **Activate the “Receive callback” workflow first**  
   Turn the workflow **on** (toggle in n8n). The production webhook is only registered when the workflow is **active**.

3. **Get the exact Production webhook URL**  
   Open the **“Video to PDF – Receive callback”** workflow, click the **Webhook (callback)** node, and **copy the Production URL** shown in the node (do not type it).  
   - Use the **Production** URL, not the Test URL.  
   - The path may be `/webhook/pipeline-done` or another path; n8n Cloud shows the real URL in the node.  
   - If you get **404** when FastAPI calls the callback, the URL is wrong or the workflow is not activated—see **“Callback returns 404”** below.

4. **In the “Submit” workflow**, open **“Set video_url, base_url, callback_url”**:
   - **base_url**: your tunnel URL (ngrok or Cloudflare), no trailing slash, e.g.  
     `https://prestricken-eugene-shinily.ngrok-free.dev`
   - **callback_url**: paste the **exact** Production URL you copied from the Receive callback workflow’s Webhook node.

5. **Activate the “Submit” workflow.**  
   When you call the trigger webhook with `{"video_url": "https://..."}`:
   - You get an immediate response: `{"job_id": "...", "message": "...", "download_url": "https://your-tunnel.../job/xxx/result"}`.
   - **Anyone** with `download_url` can open it in a browser (once the job is done) to download the PDF—no n8n access needed. Share the link with users.
   - When the pipeline finishes, FastAPI also POSTs to your callback webhook (for n8n to process the PDF). The job stays available at `download_url` until someone downloads it.

### Callback returns 404 (Not Found)

If the FastAPI terminal shows **`[callback] Failed to POST to ... : HTTP Error 404: Not Found`**:

1. **Use the exact URL from n8n**  
   In the **“Video to PDF – Receive callback (PDF when done)”** workflow, click the **Webhook (callback)** node. At the top you’ll see **Production URL** and **Test URL**. Copy the **Production URL** exactly (e.g. `https://taraztechnologies.app.n8n.cloud/webhook/XXXXX`) and set that as **callback_url** in the Submit workflow. Do not guess or type the URL; n8n Cloud may use a path that’s not exactly `/webhook/pipeline-done`.

2. **Make sure the Receive callback workflow is active**  
   In the workflow list, the “Video to PDF – Receive callback” workflow must be **on** (toggle enabled). If it’s off, the production webhook is not registered and n8n returns 404.

3. **Save the Receive callback workflow**  
   After any change, save the workflow and ensure it’s still activated.

---

## Option B: Polling workflow + increase n8n execution timeout

If you want the **same** webhook request to eventually respond with the PDF (instead of getting the PDF via callback), use the **async polling** workflow and **increase n8n’s execution timeout** so the workflow can run for the full pipeline duration (e.g. 1–2 hours).

### “Task execution timed out after 60 seconds”

n8n is killing the run because the default **execution timeout** is 60 seconds. The pipeline runs on the FastAPI side for as long as the video requires; the n8n workflow that polls is one long execution and must be allowed to run that long.

**What to do:**

1. **Per workflow (recommended)**  
   - Open the workflow that does **Submit → Poll until done → Get PDF**.  
   - Click the **⋯** menu (top right) → **Settings** (or **Workflow settings**).  
   - Enable **“Timeout”** / **“Execution timeout”** and set it to **2 hours** (or longer) so the poll loop can run until the job is done.  
   - Save.  
   - *Note:* On **n8n Cloud**, max timeout depends on your plan; if you cannot set 1–2 hours, use **Option A (callback)** instead.

2. **Self‑hosted n8n (environment variables)**  
   - `EXECUTIONS_TIMEOUT=3600` (max execution time in seconds, e.g. 1 hour).  
   - `EXECUTIONS_TIMEOUT_MAX=7200` (e.g. 2 hours).  
   - Restart n8n.

3. **Use the async polling workflow**  
   - Import **`workflow-async.json`**.  
   - Set **base_url** in the first Set node to your tunnel URL (no trailing slash).  
   - The workflow: **Submit** → **Poll** every 5s until done → **Get PDF** → **Respond**. Each *HTTP request* is short (good for Cloudflare/ngrok), but the *n8n execution* runs until the job is done, so the timeout above must be high enough.

---

## Fix in n8n (single-request workflow: “Call Pipeline API” node)

If you keep using the **single-request** workflow (`workflow.json`), do the following. Note: a single long request can still hit Cloudflare’s 100s timeout or ngrok SSL issues; prefer the async workflow above.

Your FastAPI may be at:

`https://prestricken-eugene-shinily.ngrok-free.dev/process-youtube`

Errors like **“decryption failed or bad record mac”** or weird HTML instead of a PDF usually come from **ngrok’s free-tier interstitial**. Automation (n8n) must send a special header so ngrok forwards the request to your API instead of the warning page.

---

## Fix in n8n (do this in the “Call Pipeline API” node)

### 1. Open the “Call Pipeline API” node

In your **YouTube to PDF** workflow, click the **Call Pipeline API** HTTP Request node.

### 2. Set the URL

In **URL**, use your ngrok URL **including** `/process-youtube`:

```
https://prestricken-eugene-shinily.ngrok-free.dev/process-youtube
```

(If your ngrok URL changes, update it here.)

### 3. Add the ngrok bypass header

- Find **“Send Headers”** or **“Header Parameters”** / **“Headers”**.
- Turn it **on** (e.g. **“Add option” → “Headers”** or **“Send Headers”**).
- Add one header:
  - **Name:** `Ngrok-Skip-Browser-Warning`
  - **Value:** `69420`

This tells ngrok to skip the browser warning and forward the request to your FastAPI.

### 4. (Optional) Allow self‑signed / bad certs

If you still see SSL errors:

- Open **“Options”** (or **“Add option”**).
- Enable **“Allow Unauthorized Certificates”** or **“Ignore SSL Issues”**.

Use only for testing; in production use proper HTTPS and valid certs.

### 5. Save and run

Save the workflow and run it again (e.g. via the webhook with `{"youtube_url": "https://www.youtube.com/watch?v=..."}`).

---

## “Connection closed unexpectedly” when using **HTTP** ngrok

If you run ngrok with **`--scheme=http`** and set the Call Pipeline API URL to **`http://...ngrok-free.dev/process-youtube`**, n8n Cloud may show:

**“The connection to the server was closed unexpectedly, perhaps it is offline.”**

**Reason:** n8n Cloud often allows only **HTTPS** outbound calls. Plain HTTP to your ngrok URL can be blocked or dropped, so the connection closes.

**Fix:** Use **HTTPS** instead of HTTP:

1. **Start ngrok with HTTPS** (default; do **not** use `--scheme=http`):
   ```bash
   ngrok http 8000
   ```
   You’ll get a URL like `https://prestricken-eugene-shinily.ngrok-free.dev`.

2. **In n8n**, in the **Call Pipeline API** node:
   - **URL:** `https://prestricken-eugene-shinily.ngrok-free.dev/process-youtube` (use **https**).
   - **Header:** `Ngrok-Skip-Browser-Warning` = `69420`.
   - **Options:** enable **“Allow Unauthorized Certificates”** (avoids TLS/SSL errors).

3. Save and run the workflow again.

So: **HTTP ngrok + n8n Cloud → connection closed**. **HTTPS ngrok + Allow Unauthorized Certificates → usually works.**

---

## Checklist

| Step | What to set |
|------|------------------|
| **URL** | `https://prestricken-eugene-shinily.ngrok-free.dev/process-youtube` |
| **Header** | `Ngrok-Skip-Browser-Warning` = `69420` |
| **Method** | POST |
| **Body** | JSON: `{"youtube_url": "{{ $json.youtube_url }}"}` (or from previous node) |
| **Response** | Format: **File**, Output property: e.g. **data** |
| **Timeout** | 600000 (10 min) |
| **(Optional)** | Allow Unauthorized Certificates = **true** |

---

## Why this works

- **n8n Cloud** runs on Google’s servers, so it cannot use `http://localhost:8000`. It must call a **public URL** (your ngrok URL).
- **ngrok free** shows a “Visit Site” interstitial for browser traffic. Automated requests (like n8n’s) that don’t send the bypass header can get that HTML or a mangled response, which often shows up as SSL/decryption errors.
- Sending **`Ngrok-Skip-Browser-Warning: 69420`** makes ngrok treat the request as non‑browser and forward it to your FastAPI, so n8n gets the real PDF response.

See: [How to Bypass Ngrok Browser Warning](https://stackoverflow.com/questions/73017353/how-to-bypass-ngrok-browser-warning), [ngrok free plan limits](https://ngrok.com/docs/pricing-limits/free-plan-limits/).

---

## If you still get “decryption failed or bad record mac”

That TLS error often happens when **n8n Cloud** and **ngrok** disagree on the connection (especially with large PDF responses). Try these in order.

### Option A: Use Cloudflare Tunnel instead of ngrok

Cloudflare’s quick tunnel usually works more reliably with n8n Cloud and avoids the interstitial.

1. **Install Cloudflare’s tunnel client** (one-time):
   - **Windows:**  
     [Download cloudflared](https://github.com/cloudflare/cloudflared/releases) and put `cloudflared.exe` on your PATH, or use `winget install cloudflare.cloudflared`.
   - **Or** use the portable build from the [releases page](https://github.com/cloudflare/cloudflared/releases).

2. **Start your API** (as usual):
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

3. **In another terminal**, run:
   ```bash
   cloudflared tunnel --url http://localhost:8000
   ```
   You’ll get a line like:
   ```
   https://something-random.trycloudflare.com
   ```

4. **In n8n**, in the **Call Pipeline API** node:
   - **URL:** `https://YOUR-CLOUDFLARE-URL.trycloudflare.com/process-youtube`  
     (replace with the `https://....trycloudflare.com` from the previous step.)
   - **You do not need** the `Ngrok-Skip-Browser-Warning` header for Cloudflare.
   - Under **Options**, enable **“Allow Unauthorized Certificates”** if n8n still complains about SSL.

5. Save and run the workflow again.

---

### Option B: Stay on ngrok and harden the node

1. In **Call Pipeline API**:
   - **Options** → **“Allow Unauthorized Certificates”** = **ON** (not optional if you see SSL errors).
   - Confirm **Header** `Ngrok-Skip-Browser-Warning` = `69420` is set.
   - **Timeout** at least `600000` (10 minutes).

2. Keep **ngrok** and **uvicorn** running the whole time the workflow runs.

3. If the error still appears only after the pipeline has clearly finished (logs show “Processing Complete!” / PDF generated), the failure is likely while **sending the large PDF** back. Then ngrok’s free tier may be limiting or buffering the response. In that case **Option A (Cloudflare Tunnel)** or hosting the API (e.g. Railway, Render) usually fixes it.

---

### Option C: Host the API and drop the tunnel

Deploy your FastAPI (e.g. [Railway](https://railway.app), [Render](https://render.com), [Fly.io](https://fly.io)) and use the app’s **HTTPS** URL in the **Call Pipeline API** node. No tunnel = no ngrok/Cloudflare TLS in the middle, so “bad record mac” from the tunnel goes away.

---

## If the error persists (general checklist)

1. **ngrok must be running** on your PC and tunneling to `http://localhost:8000` while you test.
2. **FastAPI** must be running: `uvicorn app:app --host 0.0.0.0 --port 8000`.
3. **URL** in n8n must be exactly:  
   `https://your-tunnel-host/process-youtube`  
   (no trailing slash, correct path).
4. **Header** `Ngrok-Skip-Browser-Warning: 69420` must be present when using **ngrok**.
5. **Options** → **“Allow Unauthorized Certificates”** = **ON** when testing.
6. If you get a new tunnel URL (e.g. after restart), update the URL in the n8n node.
