import React from "react";
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Linking,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";

import { COLORS, FONTS, SPACING, RISK } from "../constants/theme";

const SPF_HOTLINE = "tel:18007226688";
const SCAMSHIELD_URL = "https://www.scamshield.org.sg";

export default function ResultScreen({ route, navigation }) {
  const { result } = route.params;
  const level = result.risk_level || "low";
  const config = RISK[level];
  const pct = Math.round((result.risk_score ?? 0) * 100);

  return (
    <ScrollView
      style={[styles.container, { backgroundColor: config.bg }]}
      contentContainerStyle={styles.content}
    >
      {/* Header badge */}
      {level === "high" && (
        <View style={styles.alertBanner}>
          <Text style={styles.alertBannerText}>SCAM DETECTED</Text>
        </View>
      )}

      <Ionicons name={config.icon} size={64} color={config.color} />

      <Text style={[styles.title, { color: config.color }]}>
        {config.title}
      </Text>

      {/* Risk score bar */}
      <View style={styles.barTrack}>
        <View
          style={[
            styles.barFill,
            { width: `${pct}%`, backgroundColor: config.color },
          ]}
        />
      </View>
      <Text style={styles.pct}>Risk Score: {pct}%</Text>

      {/* Message */}
      <View style={styles.messageBox}>
        <Text style={styles.messageText}>{config.message}</Text>
      </View>

      {/* Reasons */}
      {result.reasons?.length > 0 && (
        <View style={styles.reasonsBox}>
          <Text style={styles.reasonsTitle}>
            {level === "high" ? "Red flags found:" : "Signals detected:"}
          </Text>
          {result.reasons.map((r, i) => (
            <View key={i} style={styles.reasonRow}>
              <Ionicons
                name={level === "high" ? "flag" : "alert-circle-outline"}
                size={16}
                color={config.color}
              />
              <Text style={styles.reasonText}>{r}</Text>
            </View>
          ))}
        </View>
      )}

      {/* Actions */}
      {level === "high" && (
        <TouchableOpacity
          style={[styles.actionBtn, { backgroundColor: COLORS.red }]}
          onPress={() => Linking.openURL(SPF_HOTLINE)}
        >
          <Ionicons name="call" size={20} color={COLORS.white} />
          <Text style={styles.actionBtnText}>
            Call Anti-Scam Hotline 1800-722-6688
          </Text>
        </TouchableOpacity>
      )}

      {level !== "low" && (
        <TouchableOpacity
          style={[styles.actionBtn, { backgroundColor: COLORS.navy }]}
          onPress={() => Linking.openURL(SCAMSHIELD_URL)}
        >
          <Ionicons name="shield" size={20} color={COLORS.white} />
          <Text style={styles.actionBtnText}>Report to ScamShield</Text>
        </TouchableOpacity>
      )}

      <TouchableOpacity
        style={[styles.actionBtn, styles.outlineBtn]}
        onPress={() => navigation.popToTop()}
      >
        <Ionicons name="scan-circle-outline" size={20} color={COLORS.navy} />
        <Text style={[styles.actionBtnText, { color: COLORS.navy }]}>
          Scan Another Image
        </Text>
      </TouchableOpacity>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1 },
  content: { padding: SPACING.lg, alignItems: "center" },

  alertBanner: {
    backgroundColor: COLORS.red,
    paddingVertical: SPACING.sm,
    paddingHorizontal: SPACING.lg,
    borderRadius: 8,
    marginBottom: SPACING.md,
  },
  alertBannerText: {
    color: COLORS.white,
    fontWeight: "800",
    fontSize: 18,
    letterSpacing: 1,
  },

  title: { ...FONTS.title, marginTop: SPACING.sm, textAlign: "center" },

  barTrack: {
    width: "100%",
    height: 10,
    backgroundColor: COLORS.greyLight,
    borderRadius: 5,
    marginTop: SPACING.lg,
    overflow: "hidden",
  },
  barFill: { height: "100%", borderRadius: 5 },
  pct: { ...FONTS.caption, marginTop: SPACING.xs },

  messageBox: {
    backgroundColor: COLORS.white,
    borderRadius: 12,
    padding: SPACING.md,
    marginTop: SPACING.lg,
    width: "100%",
  },
  messageText: { ...FONTS.regular, lineHeight: 24 },

  reasonsBox: {
    width: "100%",
    marginTop: SPACING.md,
  },
  reasonsTitle: { ...FONTS.subtitle, marginBottom: SPACING.sm },
  reasonRow: {
    flexDirection: "row",
    alignItems: "flex-start",
    gap: SPACING.sm,
    marginBottom: SPACING.xs,
  },
  reasonText: { ...FONTS.regular, flex: 1, fontSize: 14 },

  actionBtn: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: SPACING.sm,
    width: "100%",
    paddingVertical: 14,
    borderRadius: 14,
    marginTop: SPACING.md,
  },
  actionBtnText: { color: COLORS.white, fontWeight: "700", fontSize: 15 },
  outlineBtn: {
    backgroundColor: COLORS.white,
    borderWidth: 1.5,
    borderColor: COLORS.navy,
  },
});
