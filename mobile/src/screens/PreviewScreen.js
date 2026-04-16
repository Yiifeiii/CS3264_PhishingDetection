import React from "react";
import { View, Text, Image, StyleSheet, TouchableOpacity } from "react-native";
import { Ionicons } from "@expo/vector-icons";

import { COLORS, FONTS, SPACING } from "../constants/theme";

export default function PreviewScreen({ route, navigation }) {
  const { imageUri } = route.params;

  const handleAnalyse = () => {
    navigation.replace("Analysing", { imageUri });
  };

  return (
    <View style={styles.container}>
      <Text style={styles.heading}>Review your image</Text>

      <View style={styles.imageWrapper}>
        <Image
          source={{ uri: imageUri }}
          style={styles.image}
          resizeMode="contain"
        />
      </View>

      <TouchableOpacity style={styles.analyseBtn} onPress={handleAnalyse}>
        <Ionicons name="search" size={22} color={COLORS.white} />
        <Text style={styles.analyseBtnText}>Analyse Image</Text>
      </TouchableOpacity>

      <View style={styles.privacyRow}>
        <Ionicons name="lock-closed-outline" size={14} color={COLORS.grey} />
        <Text style={styles.privacyText}>
          Your image is processed securely and not stored on our servers.
        </Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.offWhite,
    padding: SPACING.lg,
  },
  heading: { ...FONTS.subtitle, marginBottom: SPACING.md },

  imageWrapper: {
    flex: 1,
    backgroundColor: COLORS.white,
    borderRadius: 12,
    overflow: "hidden",
    marginBottom: SPACING.lg,
  },
  image: { width: "100%", height: "100%" },

  analyseBtn: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: SPACING.sm,
    backgroundColor: COLORS.navy,
    paddingVertical: 16,
    borderRadius: 14,
    marginBottom: SPACING.md,
  },
  analyseBtnText: {
    color: COLORS.white,
    fontWeight: "700",
    fontSize: 17,
  },

  privacyRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: SPACING.xs,
    marginBottom: SPACING.md,
  },
  privacyText: { ...FONTS.caption, textAlign: "center" },
});
