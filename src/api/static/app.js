const sampleText =
  "58-year-old male with substernal chest pain radiating to the left arm. EKG shows ST elevation. Takes aspirin and atorvastatin.";

const statusPill = document.querySelector("#status-pill");
const triageButton = document.querySelector("#triage-button");
const sampleButton = document.querySelector("#sample-button");
const clearButton = document.querySelector("#clear-button");
const noteInput = document.querySelector("#note");
const specialtyInput = document.querySelector("#specialty");
const latency = document.querySelector("#latency");
const riskLevel = document.querySelector("#risk-level");
const riskScore = document.querySelector("#risk-score");
const validation = document.querySelector("#validation");
const reasoning = document.querySelector("#reasoning");
const carePathway = document.querySelector("#care-pathway");
const entities = document.querySelector("#entities");

async function checkHealth() {
  try {
    const response = await fetch("/health");
    if (!response.ok) throw new Error("Health check failed");
    statusPill.textContent = "API Online";
    statusPill.className = "status-pill ok";
  } catch {
    statusPill.textContent = "API Offline";
    statusPill.className = "status-pill error";
  }
}

function setLoading(isLoading) {
  triageButton.disabled = isLoading;
  triageButton.textContent = isLoading ? "Running..." : "Run Triage";
  latency.textContent = isLoading ? "Processing" : "Ready";
}

function resetResult() {
  riskLevel.textContent = "-";
  riskScore.textContent = "-";
  validation.textContent = "-";
  reasoning.textContent = "Run a note to see the assessment.";
  carePathway.textContent = "-";
  entities.innerHTML = "";
  latency.textContent = "Ready";
  document.querySelectorAll(".metric").forEach((metric) => {
    metric.className = "metric";
  });
}

function setRiskStyle(level) {
  const metric = riskLevel.closest(".metric");
  metric.className = `metric ${String(level || "").toLowerCase()}`;
}

function renderEntities(entityData) {
  entities.innerHTML = "";
  const groups = Object.entries(entityData || {});

  if (!groups.length) {
    entities.innerHTML = '<p class="empty">No entities returned.</p>';
    return;
  }

  for (const [name, values] of groups) {
    const group = document.createElement("div");
    group.className = "entity-group";

    const title = document.createElement("h3");
    title.textContent = name.replaceAll("_", " ");
    group.appendChild(title);

    const chips = document.createElement("div");
    chips.className = "chips";

    if (Array.isArray(values) && values.length) {
      for (const value of values) {
        const chip = document.createElement("span");
        chip.className = "chip";
        chip.textContent = value;
        chips.appendChild(chip);
      }
    } else {
      const empty = document.createElement("span");
      empty.className = "empty";
      empty.textContent = "None";
      chips.appendChild(empty);
    }

    group.appendChild(chips);
    entities.appendChild(group);
  }
}

async function runTriage() {
  const text = noteInput.value.trim();
  if (text.length < 20) {
    reasoning.textContent = "Enter at least 20 characters before running triage.";
    return;
  }

  setLoading(true);

  try {
    const response = await fetch("/triage", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        specialty: specialtyInput.value,
      }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Triage request failed");
    }

    riskLevel.textContent = data.risk_level || "-";
    riskScore.textContent = data.risk_score ?? "-";
    validation.textContent = data.validation_notes || "-";
    reasoning.textContent = data.risk_reasoning || "-";
    carePathway.textContent = data.care_pathway || "-";
    latency.textContent = `${data.latency_ms} ms`;
    setRiskStyle(data.risk_level);
    renderEntities(data.entities);
  } catch (error) {
    reasoning.textContent = error.message;
    carePathway.textContent = "-";
  } finally {
    setLoading(false);
  }
}

sampleButton.addEventListener("click", () => {
  specialtyInput.value = "CARDIOLOGY";
  noteInput.value = sampleText;
});

clearButton.addEventListener("click", () => {
  noteInput.value = "";
  resetResult();
});

triageButton.addEventListener("click", runTriage);
checkHealth();
