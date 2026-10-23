const btn = document.getElementById("checkBtn");
const textarea = document.getElementById("newsText");
const logLines = document.getElementById("logLines");
const emptyState = document.getElementById("emptyState");
const results = document.getElementById("results");
const stamp = document.getElementById("stamp");
const stampText = document.getElementById("stampText");
const fillReal = document.getElementById("fillReal");
const fillFake = document.getElementById("fillFake");
const pctReal = document.getElementById("pctReal");
const pctFake = document.getElementById("pctFake");
const reasonsReal = document.getElementById("reasonsReal");
const reasonsFake = document.getElementById("reasonsFake");
const clockEl = document.getElementById("clock");
const imageUpload = document.getElementById("imageUpload");
const fbCorrect = document.getElementById("fbCorrect");
const fbWrong = document.getElementById("fbWrong");
const fbStatus = document.getElementById("fbStatus");
let lastResult = null;

const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;


function tickClock() {
  const now = new Date();
  const hh = String(now.getHours()).padStart(2, "0");
  const mm = String(now.getMinutes()).padStart(2, "0");
  const ss = String(now.getSeconds()).padStart(2, "0");
  clockEl.textContent = `${hh}:${mm}:${ss}`;
}
tickClock();
setInterval(tickClock, 1000);

function addLogLine(text, delay) {
  return new Promise((resolve) => {
    setTimeout(() => {
      const line = document.createElement("div");
      line.className = "log-line";
      line.textContent = "> " + text;
      logLines.appendChild(line);
      resolve();
    }, reducedMotion ? 0 : delay);
  });
}

async function runLog(steps) {
  logLines.innerHTML = "";
  for (const step of steps) {
    await addLogLine(step, 260);
  }
  Array.from(logLines.children).forEach((el) => el.classList.add("done"));
}

function fillReasonsList(ul, words) {
  ul.innerHTML = "";
  if (!words || words.length === 0) {
    const li = document.createElement("li");
    li.className = "none";
    li.textContent = "No strong signal found either way.";
    ul.appendChild(li);
    return;
  }
  words.forEach((w) => {
    const li = document.createElement("li");
    li.textContent = w;
    ul.appendChild(li);
  });
}

// --- feedback (human-in-the-loop) ---
function resetFeedback() {
  if (fbStatus) fbStatus.textContent = "";
  if (fbCorrect) fbCorrect.disabled = false;
  if (fbWrong) fbWrong.disabled = false;
}

async function sendFeedback(isCorrect) {
  if (!lastResult) return;

  const opposite = lastResult.prediction === "FAKE" ? "REAL" : "FAKE";
  const trueLabel = isCorrect ? lastResult.prediction : opposite;

  if (fbCorrect) fbCorrect.disabled = true;
  if (fbWrong) fbWrong.disabled = true;

  try {
    const res = await fetch("/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: lastResult.text,
        prediction: lastResult.prediction,
        true_label: trueLabel,
      }),
    });
    const data = await res.json();
    if (fbStatus) {
      fbStatus.textContent = data.error
        ? data.error
        : "Saved. This will be used the next time the model is retrained.";
    }
  } catch (err) {
    console.error("[feedback] request failed:", err);
    if (fbStatus) fbStatus.textContent = "Couldn't save feedback. Try again.";
    if (fbCorrect) fbCorrect.disabled = false;
    if (fbWrong) fbWrong.disabled = false;
  }
}

fbCorrect?.addEventListener("click", () => sendFeedback(true));
fbWrong?.addEventListener("click", () => sendFeedback(false));

btn.addEventListener("click", async () => {
  const text = textarea.value.trim();
  if (!text) return;

  btn.disabled = true;
  btn.textContent = "Checking...";
  results.classList.add("hidden");
  emptyState.classList.add("hidden");
  stamp.classList.remove("stamped", "fake");
  fillReal.style.width = "0%";
  fillFake.style.width = "0%";
  lastResult = null;

  await runLog([
    "receiving copy...",
    "cross-referencing style markers...",
    "querying model...",
  ]);

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();

    if (data.error) {
      await addLogLine("error: " + data.error, 0);
      emptyState.classList.remove("hidden");
      emptyState.querySelector("p").textContent = data.error;
    } else {
      const isFake = data.prediction === "FAKE";

      lastResult = { text, prediction: data.prediction };
      resetFeedback();

      await addLogLine(
        "verdict logged: " + (isFake ? "hold — unverified" : "cleared for print"),
        0
      );

      stampText.textContent = isFake ? "HOLD" : "CLEARED";
      stamp.classList.toggle("fake", isFake);

      const realPct = data.scores?.REAL ?? 0;
      const fakePct = data.scores?.FAKE ?? 0;
      pctReal.textContent = `${realPct}%`;
      pctFake.textContent = `${fakePct}%`;

      fillReasonsList(reasonsReal, data.reasons?.REAL);
      fillReasonsList(reasonsFake, data.reasons?.FAKE);

      results.classList.remove("hidden");
      requestAnimationFrame(() => {
        stamp.classList.add("stamped");
        fillReal.style.width = `${realPct}%`;
        fillFake.style.width = `${fakePct}%`;
      });
    }
  } catch (err) {
    await addLogLine("error contacting server.", 0);
    emptyState.classList.remove("hidden");
  }

  btn.disabled = false;
  btn.textContent = "Run verification";
});

function setPasteStatus(msg) {
  const el = document.getElementById("uploadStatus");
  if (el) {
    el.textContent = msg;
  } else {
    console.log("[paste-ocr]", msg);
  }
}

async function extractImageText(file, source) {
  if (typeof Tesseract === "undefined") {
    console.error("[image-ocr] Tesseract is not defined — check the script tag in index.html");
    setPasteStatus("OCR library not loaded — refresh and try again.");
    return;
  }

  setPasteStatus(`reading ${source}...`);

  try {
    const { data } = await Tesseract.recognize(file, "eng", {
      logger: (m) => {
        if (m.status === "recognizing text") {
          setPasteStatus(`extracting text... ${Math.round(m.progress * 100)}%`);
        }
      },
    });

    const extracted = data.text.trim();
    if (!extracted) {
      setPasteStatus("no readable text found in that image.");
      return;
    }

    textarea.value = extracted;
    setPasteStatus("text extracted — review it before running verification.");
    textarea.dispatchEvent(new Event("input"));
  } catch (err) {
    console.error("[image-ocr] Tesseract.recognize failed:", err);
    setPasteStatus("couldn't read that image — try a clearer screenshot.");
  }
}

imageUpload?.addEventListener("change", async () => {
  const file = imageUpload.files?.[0];
  if (!file) return;

  await extractImageText(file, "uploaded screenshot");
  imageUpload.value = "";
});

document.addEventListener("paste", async (e) => {
  if (document.activeElement !== textarea) return;

  const items = e.clipboardData?.items;
  if (!items) return;

  const imageItem = Array.from(items).find((item) => item.type.startsWith("image/"));
  if (!imageItem) return; // no image on the clipboard — let normal text paste proceed

  e.preventDefault();

  const file = imageItem.getAsFile();
  if (!file) {
    console.error("[paste-ocr] clipboard item matched image/* but getAsFile() returned null");
    setPasteStatus("couldn't read that pasted image.");
    return;
  }

  await extractImageText(file, "pasted image");
});