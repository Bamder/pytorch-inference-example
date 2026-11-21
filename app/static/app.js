const pad = document.getElementById("pad");
const ctx = pad.getContext("2d");
const digitEl = document.getElementById("digit");
const detailEl = document.getElementById("detail");

function reset() {
  ctx.fillStyle = "#23201b";
  ctx.fillRect(0, 0, pad.width, pad.height);
  ctx.strokeStyle = "#fff";
  ctx.lineWidth = 20;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  digitEl.textContent = "?";
  detailEl.textContent = "Draw a digit from 0 to 9 on the black canvas";
}

let drawing = false;
let last = null;

// Convert pointer coordinates to canvas coordinates
function point(event) {
  const rect = pad.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) * pad.width / rect.width,
    y: (event.clientY - rect.top) * pad.height / rect.height,
  };
}

pad.addEventListener("pointerdown", (event) => {
  drawing = true;
  last = point(event);
  // A single click should also leave a dot, otherwise "1" may not be drawn
  ctx.beginPath();
  ctx.arc(last.x, last.y, ctx.lineWidth / 2, 0, Math.PI * 2);
  ctx.fillStyle = "#fff";
  ctx.fill();
  pad.setPointerCapture(event.pointerId);
});

pad.addEventListener("pointermove", (event) => {
  if (!drawing) return;
  const p = point(event);
  ctx.beginPath();
  ctx.moveTo(last.x, last.y);
  ctx.lineTo(p.x, p.y);
  ctx.stroke();
  last = p;
});

pad.addEventListener("pointerup", () => { drawing = false; });
pad.addEventListener("pointercancel", () => { drawing = false; });

document.getElementById("clear").onclick = reset;

document.getElementById("predict").onclick = async () => {
  detailEl.textContent = "Recognizing…";

  const response = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image_base64: pad.toDataURL("image/png") }),
  });
  const data = await response.json();

  if (!response.ok) {
    digitEl.textContent = "?";
    detailEl.textContent = data.detail;
    return;
  }

  digitEl.textContent = data.digit;
  detailEl.textContent = data.top3
    .map((item) => `${item.digit}: ${(item.score * 100).toFixed(1)}%`)
    .join("  |  ");
};

reset();
