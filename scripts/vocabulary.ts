// Candidate names for clusters and per-image tags, scored zero-shot against
// the images with the SigLIP2 text tower (see scripts/space-build.ts).
//
// SPECIFIC names fine clusters and tags images: design categories, materials
// and aesthetics that a moodboard is organized by. Narrow object nouns (fish,
// sports cars, chairs) are left out on purpose: SigLIP fires on them loosely,
// and they rarely describe a group of images.
//
// BROAD names the top-level groups.

export const SPECIFIC: string[] = [
  // interfaces
  "app screens", "phone mockups", "mobile UI", "app onboarding", "dark UI", "light UI", "dashboards",
  "web pages", "landing pages", "design systems", "UI components", "buttons", "glass buttons",
  "toggles", "app icons", "pixel icons", "widgets", "lock screens", "music player", "media player",
  "chat interface", "settings screen", "browser window", "desktop UI", "code editor", "terminal",
  "spreadsheets", "charts", "data visualization", "infographics", "wireframes", "whiteboard",
  "sticky notes", "cards UI", "article page", "portfolio site", "ecommerce", "AI interface",
  "game UI", "retro UI", "Y2K UI", "skeuomorphic UI", "aqua buttons", "frutiger aero",
  "glassmorphism", "neumorphism", "flat design", "brutalist web", "minimal UI", "soft UI",
  "bento grid", "interface design", "product UI", "photo gallery", "social media",
  // objects
  "iPod", "phones", "phone cases", "headphones", "speakers", "radios", "walkman", "cassettes",
  "vinyl records", "cameras", "keyboards", "keycaps", "retro computers", "laptops", "televisions",
  "game consoles", "watches", "clocks", "calculators", "synthesizers", "audio gear", "gadgets",
  "consumer electronics", "devices", "hardware", "industrial design", "product design",
  "product renders", "3D renders", "chrome objects", "metal objects", "glass objects",
  "plastic objects", "toys", "jewelry", "rings", "sneakers", "bags", "jackets", "leather goods",
  "fashion", "furniture", "lamps", "ceramics", "bottles", "cans", "cosmetics", "packaging",
  "labels", "stationery", "business cards", "envelopes", "postcards", "stamps", "tickets",
  "credit cards", "cars", "car renders", "robots", "mechanical parts", "control panels", "dials",
  "switches", "pills", "everyday objects",
  // graphic design
  "typography", "type posters", "lettering", "calligraphy", "type specimens", "wordmarks", "logos",
  "monograms", "brand identity", "visual identity", "symbols", "pictograms", "pixel type",
  "bitmap fonts", "dot matrix", "ASCII art", "glitch type", "chrome type", "3D type",
  "swiss style", "grid systems", "bauhaus", "modernism", "brutalism", "maximalism", "minimalism",
  "editorial design", "magazine layouts", "posters", "brochures", "books", "book covers",
  "magazines", "zines", "print design", "signage", "wayfinding", "stickers", "badges", "collage",
  "cut paper", "risograph", "screen print", "halftone", "duotone", "gradients", "mesh gradients",
  "noise texture", "film grain", "patterns", "geometric patterns", "abstract shapes", "blobs",
  "organic shapes", "3D shapes", "isometric", "illustration", "line art", "vector art",
  "character design", "mascots", "cartoons", "comics", "emoji", "smiley faces", "doodles",
  "sketches", "paintings", "watercolor", "oil painting", "ink drawing", "japanese art",
  "pixel art", "generative art", "glitch art", "album covers", "event posters", "invitations",
  "presentation slides", "motion graphics",
  // photography and subjects
  "photography", "film photos", "analog photos", "polaroids", "black and white photos",
  "portraits", "faces", "eyes", "close-ups", "hands", "hand holding phone", "people",
  "silhouettes", "fashion photography", "editorial photography", "still life",
  "product photography", "studio shots", "flash photography", "motion blur", "long exposure",
  "blurry photos", "soft focus", "bokeh", "light leaks", "lens flare", "glow", "neon lights",
  "light trails", "cinematic", "moody", "dark scenes", "night", "landscapes", "green hills",
  "fields", "grass", "forests", "plants", "leaves", "moss", "flowers", "petals", "mountains",
  "snow", "ocean", "waves", "nature", "rocks", "stones", "sand", "sky", "clouds", "sunsets",
  "golden hour", "outer space", "planets", "eclipse", "black hole", "stars", "architecture",
  "buildings", "interiors", "rooms", "cities", "streets", "animals", "food", "drinks", "water",
  "liquid", "smoke", "light and shadow", "reflections",
  // materials
  "chrome", "metal", "silver", "gold", "brushed metal", "liquid metal", "glass", "frosted glass",
  "crystal", "plastic", "rubber", "leather", "fabric", "knit", "paper", "cardboard", "embossed",
  "debossed", "engraved", "letterpress", "foil", "wood", "marble", "concrete", "clay", "ceramic",
  "wax", "foam", "skin", "translucent", "transparent", "reflective", "holographic", "iridescent",
  "matte", "glossy", "textured", "tactile", "inflated", "melted", "crumpled", "torn paper",
  // aesthetics
  "minimal", "clean", "elegant", "luxury", "playful", "quirky", "whimsical", "cute", "bold",
  "quiet", "calm", "dreamy", "ethereal", "surreal", "abstract", "conceptual", "experimental",
  "avant-garde", "raw", "gritty", "grunge", "distressed", "weathered", "organic", "natural",
  "earthy", "technical", "engineered", "industrial", "utilitarian", "scientific", "sleek",
  "futuristic", "high tech", "lo-fi", "analog", "digital", "glitchy", "pixelated", "generative",
  "sci-fi", "cyberpunk", "retro futurism", "vaporwave", "Y2K", "90s", "80s", "vintage", "retro",
  "nostalgic", "archival", "monochrome", "black and white", "pastel", "soft colors",
  "muted colors", "earth tones", "vibrant colors", "neon", "colorful", "rainbow", "dark", "bright",
  "airy", "warm tones", "cool tones", "red accents", "orange", "yellow", "green", "blue",
  "purple", "pink", "beige", "grayscale", "high contrast", "spatial design", "apple style",
  "japanese minimalism", "scandinavian", "mid-century", "art deco", "streetwear", "techwear",
  "editorial", "corporate", "fintech", "crypto",
];

export const BROAD: string[] = [
  "interfaces", "web design", "mobile apps", "software", "objects", "products", "hardware",
  "industrial design", "typography", "print", "graphic design", "branding", "illustration",
  "art", "painting", "photography", "nature", "landscapes", "light", "color", "space", "people",
  "portraits", "fashion", "architecture", "interiors", "abstract", "3D", "materials", "textures",
  "packaging", "stationery", "pixel art", "digital art", "motion",
  "retro tech", "animals", "food", "cars", "technology", "editorial", "collage", "patterns",
  "shapes", "tactile", "screens", "phones", "music", "games", "data", "code", "plants", "film",
  "devices", "minimalism", "sci-fi", "nostalgia",
];
