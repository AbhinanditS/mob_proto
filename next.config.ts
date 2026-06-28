import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // @huggingface/transformers is used ONLY client-side (the text-query
  // embedding runs in a browser click handler, never during SSR/prerender),
  // but it pulls in Node-only deps (onnxruntime-node, sharp) whose native
  // binaries total ~375MB. Two separate Next mechanisms would otherwise
  // include them and break the Vercel build:
  //   1) webpack tries to bundle them into the browser build — alias to false
  //      so the browser uses onnxruntime-web instead.
  //   2) Output File Tracing follows the dynamic-import edge and would copy
  //      the native binaries into the serverless function, exceeding Vercel's
  //      250MB limit — exclude them from the trace since the server never
  //      executes this code path.
  webpack: (config) => {
    config.resolve.alias = {
      ...config.resolve.alias,
      "onnxruntime-node$": false,
      "sharp$": false,
    };
    return config;
  },
  outputFileTracingExcludes: {
    "*": [
      "node_modules/@huggingface/transformers/**",
      "node_modules/onnxruntime-node/**",
      "node_modules/sharp/**",
    ],
  },
};

export default nextConfig;
