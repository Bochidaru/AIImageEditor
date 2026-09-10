export function readFileAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(reader.error ?? new Error("Failed to read file."));
    reader.readAsDataURL(file);
  });
}

export const ACCEPTED_IMAGE_TYPES = ["image/png", "image/jpeg", "image/webp"];
export const MAX_UPLOAD_MB = 25;

export function validateImageFile(file: File): string | null {
  if (!ACCEPTED_IMAGE_TYPES.includes(file.type)) {
    return "Unsupported format. Use PNG, JPEG, or WebP.";
  }
  if (file.size > MAX_UPLOAD_MB * 1024 * 1024) {
    return `File is too large. Maximum size is ${MAX_UPLOAD_MB}MB.`;
  }
  return null;
}
