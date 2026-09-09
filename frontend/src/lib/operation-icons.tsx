import {
  Eraser,
  ArrowsOutCardinal,
  Image as ImageIcon,
  MagicWand,
  PaintBrush,
  Sparkle,
  Stack,
  Swap,
  UserSwitch,
  type Icon,
} from "@phosphor-icons/react";
import type { OperationId } from "@/lib/types";

export const OPERATION_ICONS: Record<OperationId, Icon> = {
  segment: MagicWand,
  remove_object: Eraser,
  replace_object: Swap,
  replace_background: Stack,
  add_object_by_prompt: PaintBrush,
  add_object_by_reference: UserSwitch,
  prompt_edit: MagicWand,
  generate_image: Sparkle,
  outpaint: ArrowsOutCardinal,
  upscale: ImageIcon,
};
