// ScamCheck SG colour palette and layout tokens.

export const COLORS = {
  navy: "#1A2E4A",
  navyLight: "#2A4060",
  white: "#FFFFFF",
  offWhite: "#F5F6FA",
  grey: "#8E9AAF",
  greyLight: "#D1D5DE",
  black: "#111827",

  // Risk colours
  green: "#2ECC71",
  greenBg: "#EAFAF1",
  amber: "#F39C12",
  amberBg: "#FEF5E7",
  red: "#E74C3C",
  redBg: "#FDEDEC",
};

export const RISK = {
  low: {
    color: COLORS.green,
    bg: COLORS.greenBg,
    icon: "checkmark-circle",
    title: "Looks Safe!",
    message:
      "Our analysis did not detect signs of a scam. You can proceed with caution, but always trust your gut.",
  },
  medium: {
    color: COLORS.amber,
    bg: COLORS.amberBg,
    icon: "alert-circle",
    title: "Proceed with Caution",
    message:
      "We detected some suspicious signals. Do NOT click any links or share personal information.",
  },
  high: {
    color: COLORS.red,
    bg: COLORS.redBg,
    icon: "warning",
    title: "Scam Detected!",
    message:
      "HIGH CHANCE OF SCAM. Do NOT respond, click links, or transfer money. Block the sender immediately.",
  },
};

export const FONTS = {
  regular: { fontSize: 16, color: COLORS.black },
  title: { fontSize: 28, fontWeight: "700", color: COLORS.black },
  subtitle: { fontSize: 18, fontWeight: "600", color: COLORS.black },
  caption: { fontSize: 13, color: COLORS.grey },
};

export const SPACING = {
  xs: 4,
  sm: 8,
  md: 16,
  lg: 24,
  xl: 32,
};
