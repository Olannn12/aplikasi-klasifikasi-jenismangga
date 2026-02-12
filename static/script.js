// ===== GLOBAL VARIABLES =====
const uploadArea = document.getElementById("uploadArea");
const fileInput = document.getElementById("fileInput");
const previewArea = document.getElementById("previewArea");
const previewImage = document.getElementById("previewImage");
const removeBtn = document.getElementById("removeBtn");
const predictBtn = document.getElementById("predictBtn");
const btnText = document.getElementById("btnText");
const btnLoader = document.getElementById("btnLoader");
const resultsSection = document.getElementById("resultsSection");

let selectedFile = null;

// ===== UPLOAD AREA EVENTS =====
uploadArea.addEventListener("click", () => {
  fileInput.click();
});

uploadArea.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadArea.classList.add("drag-over");
});

uploadArea.addEventListener("dragleave", () => {
  uploadArea.classList.remove("drag-over");
});

uploadArea.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadArea.classList.remove("drag-over");

  const files = e.dataTransfer.files;
  if (files.length > 0) {
    handleFileSelect(files[0]);
  }
});

// ===== FILE INPUT CHANGE =====
fileInput.addEventListener("change", (e) => {
  if (e.target.files.length > 0) {
    handleFileSelect(e.target.files[0]);
  }
});

// ===== HANDLE FILE SELECTION =====
function handleFileSelect(file) {
  // Validate file type
  if (!file.type.startsWith("image/")) {
    showAlert("error", "File harus berupa gambar (PNG, JPG, atau JPEG)");
    return;
  }

  // Validate file size (max 10MB)
  if (file.size > 10 * 1024 * 1024) {
    showAlert("error", "Ukuran file maksimal 10MB");
    return;
  }

  selectedFile = file;

  // Show preview
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImage.src = e.target.result;
    uploadArea.style.display = "none";
    previewArea.style.display = "block";
    predictBtn.disabled = false;

    // Hide results if shown
    resultsSection.style.display = "none";
  };
  reader.readAsDataURL(file);
}

// ===== REMOVE IMAGE =====
removeBtn.addEventListener("click", () => {
  selectedFile = null;
  fileInput.value = "";
  previewImage.src = "";
  previewArea.style.display = "none";
  uploadArea.style.display = "block";
  predictBtn.disabled = true;
  resultsSection.style.display = "none";
});

// ===== PREDICT BUTTON =====
predictBtn.addEventListener("click", async () => {
  if (!selectedFile) {
    showAlert("error", "Pilih gambar terlebih dahulu");
    return;
  }

  // Show loading state
  predictBtn.disabled = true;
  btnText.textContent = "Menganalisis...";
  btnLoader.style.display = "inline-block";

  try {
    // Create form data
    const formData = new FormData();
    formData.append("file", selectedFile);

    // Send request to API
    const response = await fetch("/api/predict", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error("Prediction failed");
    }

    const result = await response.json();

    // Display results
    displayResults(result);

    showAlert("success", "Klasifikasi berhasil!");
  } catch (error) {
    console.error("Prediction error:", error);
    showAlert(
      "error",
      "Terjadi kesalahan saat klasifikasi. Silakan coba lagi.",
    );
  } finally {
    // Reset button state
    predictBtn.disabled = false;
    btnText.textContent = "Klasifikasi Mangga";
    btnLoader.style.display = "none";
  }
});

// ===== DISPLAY RESULTS =====
function displayResults(data) {
  if (!data.success) {
    showAlert("error", "Gagal mendapatkan hasil prediksi");
    return;
  }

  const { prediction, top3, all_probabilities } = data;

  // Main prediction
  document.getElementById("resultClass").textContent = prediction.class_name;
  document.getElementById("confidenceValue").textContent =
    prediction.confidence.toFixed(1);

  // Set confidence badge color
  const badge = document.getElementById("confidenceBadge");
  if (prediction.confidence >= 90) {
    badge.style.background = "var(--success)";
  } else if (prediction.confidence >= 70) {
    badge.style.background = "var(--warning)";
  } else {
    badge.style.background = "var(--danger)";
  }

  // Top 3 predictions
  const top3List = document.getElementById("top3List");
  top3List.innerHTML = "";

  top3.forEach((item, index) => {
    const div = document.createElement("div");
    div.className = "prediction-item";
    div.innerHTML = `
            <span class="prediction-name">${index + 1}. ${item.class_name}</span>
            <span class="prediction-confidence">${item.confidence.toFixed(1)}%</span>
        `;
    top3List.appendChild(div);
  });

  // All probabilities with bars
  const allProbabilities = document.getElementById("allProbabilities");
  allProbabilities.innerHTML = "";

  // Sort by probability
  const sortedProbs = Object.entries(all_probabilities).sort(
    (a, b) => b[1] - a[1],
  );

  sortedProbs.forEach(([className, probability]) => {
    const div = document.createElement("div");
    div.className = "probability-bar";
    div.innerHTML = `
            <div class="probability-header">
                <span class="probability-label">${className}</span>
                <span class="probability-value">${probability.toFixed(1)}%</span>
            </div>
            <div class="probability-track">
                <div class="probability-fill" style="width: ${probability}%"></div>
            </div>
        `;
    allProbabilities.appendChild(div);
  });

  // Show results section with animation
  resultsSection.style.display = "block";
  resultsSection.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ===== ALERT FUNCTION =====
function showAlert(type, message) {
  // Create alert element
  const alert = document.createElement("div");
  alert.style.cssText = `
        position: fixed;
        top: 100px;
        right: 20px;
        padding: 16px 24px;
        border-radius: 8px;
        color: white;
        font-weight: 600;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        z-index: 1000;
        animation: slideInRight 0.3s ease;
        max-width: 400px;
    `;

  // Set background based on type
  if (type === "success") {
    alert.style.background = "#10b981";
  } else if (type === "error") {
    alert.style.background = "#ef4444";
  } else {
    alert.style.background = "#3b82f6";
  }

  alert.textContent = message;
  document.body.appendChild(alert);

  // Add slide in animation
  const style = document.createElement("style");
  style.textContent = `
        @keyframes slideInRight {
            from {
                transform: translateX(400px);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }
    `;
  document.head.appendChild(style);

  // Remove after 3 seconds
  setTimeout(() => {
    alert.style.animation = "slideOutRight 0.3s ease";
    setTimeout(() => {
      document.body.removeChild(alert);
      document.head.removeChild(style);
    }, 300);
  }, 3000);

  // Add slide out animation
  style.textContent += `
        @keyframes slideOutRight {
            from {
                transform: translateX(0);
                opacity: 1;
            }
            to {
                transform: translateX(400px);
                opacity: 0;
            }
        }
    `;
}

// ===== CHECK API HEALTH ON LOAD =====
async function checkAPIHealth() {
  try {
    const response = await fetch("/api/health");
    const data = await response.json();

    if (data.status === "healthy" && data.model_loaded) {
      console.log("✓ API is healthy and model is loaded");
    } else {
      console.warn("⚠ API is running but model is not loaded");
      showAlert("error", "Model belum dimuat. Silakan restart server.");
    }
  } catch (error) {
    console.error("✗ Cannot connect to API");
    showAlert(
      "error",
      "Tidak dapat terhubung ke server. Pastikan backend berjalan.",
    );
  }
}

// Check API health when page loads
window.addEventListener("load", () => {
  checkAPIHealth();
  console.log("🥭 Mango Classification Web App Loaded");
});

// ===== KEYBOARD SHORTCUTS =====
document.addEventListener("keydown", (e) => {
  // Ctrl/Cmd + U to upload
  if ((e.ctrlKey || e.metaKey) && e.key === "u") {
    e.preventDefault();
    fileInput.click();
  }

  // Enter to predict (if image is selected)
  if (e.key === "Enter" && !predictBtn.disabled) {
    predictBtn.click();
  }

  // Escape to remove image
  if (e.key === "Escape" && selectedFile) {
    removeBtn.click();
  }
});
