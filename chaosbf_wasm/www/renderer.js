// Canvas 2D Renderer for ChaosBF visualization
export class Renderer {
  constructor(canvas, width, height) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d', { alpha: false });
    this.width = width;
    this.height = height;

    // Set canvas resolution
    canvas.width = width;
    canvas.height = height;

    // Create ImageData for frame rendering
    this.imageData = this.ctx.createImageData(width, height);

    // Color palettes
    this.palette = this.createPalette();
  }

  createPalette() {
    // Create a vibrant color palette for cell values 0-255
    const pal = new Uint8ClampedArray(256 * 4);

    for (let i = 0; i < 256; i++) {
      // HSV to RGB with smooth transitions
      const h = (i / 255.0) * 360;
      const s = 0.8;
      const v = 0.3 + (i / 255.0) * 0.7;

      const c = v * s;
      const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
      const m = v - c;

      let r, g, b;
      if (h < 60) { r = c; g = x; b = 0; }
      else if (h < 120) { r = x; g = c; b = 0; }
      else if (h < 180) { r = 0; g = c; b = x; }
      else if (h < 240) { r = 0; g = x; b = c; }
      else if (h < 300) { r = x; g = 0; b = c; }
      else { r = c; g = 0; b = x; }

      pal[i * 4 + 0] = Math.round((r + m) * 255);
      pal[i * 4 + 1] = Math.round((g + m) * 255);
      pal[i * 4 + 2] = Math.round((b + m) * 255);
      pal[i * 4 + 3] = 255;
    }

    return pal;
  }

  render(frame) {
    // Map frame values to RGBA using palette
    const pixels = this.imageData.data;

    for (let i = 0; i < frame.length; i++) {
      const val = frame[i];
      const pi = i * 4;
      const ci = val * 4;

      pixels[pi + 0] = this.palette[ci + 0];
      pixels[pi + 1] = this.palette[ci + 1];
      pixels[pi + 2] = this.palette[ci + 2];
      pixels[pi + 3] = this.palette[ci + 3];
    }

    // Draw to canvas
    this.ctx.putImageData(this.imageData, 0, 0);
  }

  resize(width, height) {
    this.width = width;
    this.height = height;
    this.canvas.width = width;
    this.canvas.height = height;
    this.imageData = this.ctx.createImageData(width, height);
  }
}
