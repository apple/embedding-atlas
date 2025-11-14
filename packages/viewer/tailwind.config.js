const defaultTheme = require("tailwindcss/defaultTheme");

const baseFontSize = 13; // px

/** Create a override config to convert all rem units to px units with the `baseFontSize` */
function resolveREM(input) {
  if (typeof input == "string") {
    if (!input.match(/^[0-9\.]+rem$/)) {
      return;
    }
    let num = +input.replace("rem", "");
    return num * baseFontSize + "px";
  } else if (typeof input == "object") {
    if (input instanceof Array) {
      let result = [];
      let hasKey = false;
      for (let item of input) {
        let r = resolveREM(item);
        if (r !== undefined) {
          result.push(r);
          hasKey = true;
        } else {
          result.push(item);
        }
      }
      if (hasKey) {
        return result;
      }
    } else {
      let result = {};
      let hasKey = false;
      for (let key in input) {
        let r = resolveREM(input[key]);
        if (r !== undefined) {
          result[key] = r;
          hasKey = true;
        }
      }
      if (hasKey) {
        return result;
      }
    }
  }
}

/** @type {import('tailwindcss').Config} */
export default {
  theme: {
    extend: resolveREM(defaultTheme),
  },
  plugins: [require("@tailwindcss/typography")],
};
