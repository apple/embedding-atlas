import { toSvg } from "html-to-image";

export interface ScreenshotOptions {
  maxWidth?: number;
  maxHeight?: number;
  pixelRatio?: number;
  backgroundColor?: string;
}

export async function screenshot(element: HTMLElement, options: ScreenshotOptions = {}): Promise<string> {
  let maxWidth = options.maxWidth ?? 16384;
  let maxHeight = options.maxHeight ?? 16384;
  let pixelRatio = options.pixelRatio ?? 1;

  let replacer = svgForeignObjectCanvasFixer(element);

  let svg = await toSvg(element, { pixelRatio: pixelRatio });

  if (replacer) {
    let content = await (await fetch(svg)).text();
    content = replacer(content);
    svg = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(content);
  }

  let img = await createImage(svg);

  let canvas = document.createElement("canvas");
  canvas.width = img.width * pixelRatio;
  canvas.height = img.height * pixelRatio;
  if (canvas.width > maxWidth || canvas.height > maxHeight) {
    // Set the canvas size to up to maxWidth x maxHeight (keep aspect ratio)
    let scaleX = maxWidth / canvas.width;
    let scaleY = maxHeight / canvas.height;
    let scale = Math.min(scaleX, scaleY);

    canvas.width = Math.floor(canvas.width * scale);
    canvas.height = Math.floor(canvas.height * scale);
  }

  let ctx = canvas.getContext("2d");
  if (ctx) {
    if (options.backgroundColor != null) {
      ctx.fillStyle = options.backgroundColor;
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  }
  return canvas.toDataURL("image/png");
}

function createImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    let img = new Image();
    img.onload = () => {
      img.decode().then(() => {
        requestAnimationFrame(() => resolve(img));
      });
    };
    img.onerror = reject;
    img.crossOrigin = "anonymous";
    img.decoding = "async";
    img.src = url;
  });
}

function svgForeignObjectCanvasFixer(root: HTMLElement): ((text: string) => string) | undefined {
  // Find all canvas elements inside SVG foreignObject elements
  let elements = root.querySelectorAll("svg foreignObject > canvas");

  let map = new Map<string, HTMLCanvasElement>();

  let i = 0;
  for (let element of elements) {
    let key = `canvas_${Math.random().toString(36).substring(2, 11)}_${i}`;
    map.set(key, element as HTMLCanvasElement);
    element.setAttribute("data-screenshot-temp", key);
    i += 1;
  }

  if (map.size == 0) {
    return undefined;
  }

  return (text) => {
    for (let [key, canvas] of map.entries()) {
      let index = text.indexOf(key);
      if (index >= 0) {
        // Pattern:
        // <foreignObject ...<canvas ... data-screenshot-temp="{key}" ... ></canvas></foreignObject>
        // Replace to:
        // <image href="{}" x={} y={} width={} height={} />
        let startingIndex = text.lastIndexOf("<foreignObject", index);
        let closingIndex = text.indexOf("</foreignObject>", index) + "</foreignObject>".length;
        // Copy the x, y, width, height from the foreignObject to the replaced image
        let p = canvas.parentElement! as Element;
        let x = p.getAttribute("x");
        let y = p.getAttribute("y");
        let w = p.getAttribute("width");
        let h = p.getAttribute("height");
        let img = `<image href="${canvas.toDataURL()}" x="${x}" y="${y}" width="${w}" height="${h}" />`;
        text = text.substring(0, startingIndex) + img + text.substring(closingIndex);
      }
    }
    return text;
  };
}
