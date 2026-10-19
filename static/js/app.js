const form = document.getElementById("uploadForm");
const input = document.getElementById("imageInput");
const dropZone = document.getElementById("dropZone");
const fileName = document.getElementById("fileName");
const resultPanel = document.getElementById("resultPanel");

input.addEventListener("change", () => {
  fileName.textContent = input.files.length
    ? input.files[0].name
    : "";
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  if (!input.files.length) {
    alert("Please select a chest X-ray image first.");
    return;
  }

  const button = form.querySelector(".analyze-button");
  button.disabled = true;
  button.innerHTML = "Analyzing…";

  const data = new FormData();
  data.append("image", input.files[0]);

  resultPanel.innerHTML = `
    <div class="result-empty">
      <div class="result-icon">◌</div>
      <h3>Analyzing X-ray…</h3>
      <p>The CNN and Grad-CAM pipeline is processing your image.</p>
    </div>`;

  try {
    const response = await fetch("/predict", { method: "POST", body: data });
    const result = await response.json();

    if (!response.ok) throw new Error(result.error || "Prediction failed.");

    const isPneumonia = result.prediction === "PNEUMONIA";
    resultPanel.innerHTML = `
      <div class="result-content">
        <span class="result-badge">${isPneumonia ? "AI RESULT" : "AI RESULT"}</span>
        <div class="result-title">${result.prediction}</div>
        <div class="confidence">Model confidence: <strong>${result.confidence}%</strong></div>
        <img class="gradcam" src="${result.gradcam}?t=${Date.now()}" alt="Grad-CAM explanation">
        <div class="result-note">
          Grad-CAM visualizes image regions that influenced the model's
          pneumonia-output prediction. It is an explainability aid, not a
          clinical diagnosis.
        </div>
      </div>`;
  } catch (error) {
    resultPanel.innerHTML = `
      <div class="result-empty">
        <div class="result-icon">!</div>
        <h3>Unable to analyze</h3>
        <p>${error.message}</p>
      </div>`;
  } finally {
    button.disabled = false;
    button.innerHTML = 'Analyze X-Ray <span>→</span>';
  }
});
