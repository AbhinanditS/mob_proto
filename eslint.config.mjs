import { FlatCompat } from "@eslint/eslintrc";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const compat = new FlatCompat({
  baseDirectory: __dirname,
});

const eslintConfig = [
  {
    // Build artifacts — linting .next output floods the report the first
    // time `npm run lint` runs after a build.
    ignores: ["node_modules/**", ".next/**", "out/**", "build/**", "next-env.d.ts"],
  },
  ...compat.extends("next/core-web-vitals"),
  ...compat.extends("next/typescript"),
  {
    rules: {
      "@next/next/no-img-element": "off",
    },
  },
];

export default eslintConfig;
