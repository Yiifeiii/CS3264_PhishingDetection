// API client for the ScamCheck SG FastAPI backend.

// When running Expo Go on a physical device, replace with your machine's LAN IP.
// e.g. "http://192.168.1.42:8000"
const BASE_URL = "http://localhost:8000";

/**
 * Upload an image to the backend and return the analysis result.
 *
 * @param {string} imageUri  Local file URI from expo-image-picker.
 * @returns {Promise<{risk_level: string, risk_score: number, image_score: number, text_score: number, reasons: string[]}>}
 */
export async function analyseImage(imageUri) {
  const filename = imageUri.split("/").pop() || "photo.jpg";
  const match = /\.(\w+)$/.exec(filename);
  const ext = match ? match[1].toLowerCase() : "jpg";
  const mimeType = `image/${ext === "jpg" ? "jpeg" : ext}`;

  const formData = new FormData();
  formData.append("image", {
    uri: imageUri,
    name: filename,
    type: mimeType,
  });

  const response = await fetch(`${BASE_URL}/api/analyse`, {
    method: "POST",
    body: formData,
    headers: { "Content-Type": "multipart/form-data" },
  });

  if (!response.ok) {
    throw new Error(`Server error: ${response.status}`);
  }

  return response.json();
}

export async function healthCheck() {
  const response = await fetch(`${BASE_URL}/api/health`);
  return response.json();
}
