/**
 * Single entry point every component should import from. Switches between
 * the real HTTP client (src/inpaint_core/api/) and the offline mock based
 * on NEXT_PUBLIC_USE_MOCK_API — set it to "true" to demo the UI without a
 * running backend. Defaults to the real client.
 */
import * as client from "@/lib/api/client";
import * as mock from "@/lib/api/mock";

const useMock = process.env.NEXT_PUBLIC_USE_MOCK_API === "true";
const impl = useMock ? mock : client;

export const registerImage = impl.registerImage;
export const segment = impl.segment;
export const removeObject = impl.removeObject;
export const replaceObject = impl.replaceObject;
export const replaceBackground = impl.replaceBackground;
export const addObjectByPrompt = impl.addObjectByPrompt;
export const addObjectByReference = impl.addObjectByReference;
export const promptEdit = impl.promptEdit;
export const generateImage = impl.generateImage;
export const outpaint = impl.outpaint;
export const upscale = impl.upscale;

export { ApiError } from "@/lib/api/errors";
