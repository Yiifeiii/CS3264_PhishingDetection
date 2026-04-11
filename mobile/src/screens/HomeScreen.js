import React, { useEffect, useState } from "react";
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
} from "react-native";
import * as ImagePicker from "expo-image-picker";
import { Ionicons } from "@expo/vector-icons";

import { COLORS, FONTS, SPACING } from "../constants/theme";
import TIPS from "../constants/tips";

export default function HomeScreen({ navigation }) {
  const [tip, setTip] = useState(TIPS[0]);

  useEffect(() => {
    setTip(TIPS[Math.floor(Math.random() * TIPS.length)]);
  }, []);

  const pickImage = async () => {
    const perm = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!perm.granted) return;
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ["images"],
      quality: 0.8,
      allowsEditing: true,
    });

    if (!result.canceled && result.assets?.[0]) {
      navigation.navigate("Preview", { imageUri: result.assets[0].uri });
    }
  };

  return (
    <ScrollView
      style={styles.container}
      contentContainerStyle={styles.content}
    >
      {/* Hero */}
      <View style={styles.hero}>
        <Ionicons name="shield-checkmark" size={64} color={COLORS.navy} />
        <Text style={styles.heroTitle}>ScamCheck SG</Text>
        <Text style={styles.heroSubtitle}>Stay safe, stay smart</Text>
      </View>

      {/* Upload zone */}
      <View style={styles.uploadZone}>
        <Ionicons name="cloud-upload-outline" size={48} color={COLORS.grey} />
        <Text style={styles.uploadText}>
          Upload a screenshot to check for scams
        </Text>

        <TouchableOpacity
          style={[styles.btn, styles.btnPrimary]}
          onPress={pickImage}
        >
          <Ionicons name="images" size={20} color={COLORS.white} />
          <Text style={styles.btnTextLight}>Upload Image</Text>
        </TouchableOpacity>
      </View>

      {/* Scam tip */}
      <View style={styles.tipCard}>
        <Ionicons name="bulb-outline" size={20} color={COLORS.amber} />
        <Text style={styles.tipText}>{tip}</Text>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: COLORS.offWhite },
  content: { padding: SPACING.lg, alignItems: "center" },

  hero: { alignItems: "center", marginBottom: SPACING.lg },
  heroTitle: { ...FONTS.title, color: COLORS.navy, marginTop: SPACING.sm },
  heroSubtitle: { ...FONTS.caption, marginTop: SPACING.xs },

  uploadZone: {
    width: "100%",
    backgroundColor: COLORS.white,
    borderRadius: 16,
    borderWidth: 2,
    borderColor: COLORS.greyLight,
    borderStyle: "dashed",
    padding: SPACING.xl,
    alignItems: "center",
    marginBottom: SPACING.lg,
  },
  uploadText: {
    ...FONTS.regular,
    color: COLORS.grey,
    textAlign: "center",
    marginVertical: SPACING.md,
  },

  btn: {
    flexDirection: "row",
    alignItems: "center",
    gap: SPACING.sm,
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 12,
  },
  btnPrimary: { backgroundColor: COLORS.navy },
  btnOutline: {
    backgroundColor: COLORS.white,
    borderWidth: 1.5,
    borderColor: COLORS.navy,
  },
  btnTextLight: { color: COLORS.white, fontWeight: "600", fontSize: 15 },
  btnTextDark: { color: COLORS.navy, fontWeight: "600", fontSize: 15 },

  tipCard: {
    width: "100%",
    flexDirection: "row",
    backgroundColor: COLORS.amberBg,
    borderRadius: 12,
    padding: SPACING.md,
    gap: SPACING.sm,
    alignItems: "flex-start",
  },
  tipText: { ...FONTS.caption, color: COLORS.black, flex: 1 },
});
