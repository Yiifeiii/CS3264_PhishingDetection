import React from "react";
import { View, Text, StyleSheet, ScrollView, Linking, TouchableOpacity } from "react-native";
import { Ionicons } from "@expo/vector-icons";

import { COLORS, FONTS, SPACING } from "../constants/theme";

export default function AboutScreen() {
  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <View style={styles.header}>
        <Ionicons name="shield-checkmark" size={48} color={COLORS.white} />
        <Text style={styles.headerTitle}>ScamCheck SG</Text>
        <Text style={styles.headerSub}>v1.0.0</Text>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>How It Works</Text>
        <Text style={styles.body}>
          ScamCheck SG uses AI to analyse screenshots for scam indicators. Upload
          a photo of a suspicious message, website, or advertisement and our model
          will assess the risk level.
        </Text>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Useful Resources</Text>

        <TouchableOpacity
          style={styles.link}
          onPress={() => Linking.openURL("https://www.scamshield.org.sg")}
        >
          <Ionicons name="shield-outline" size={20} color={COLORS.navy} />
          <Text style={styles.linkText}>ScamShield (Report scams)</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.link}
          onPress={() => Linking.openURL("https://www.scamalert.sg")}
        >
          <Ionicons name="alert-circle-outline" size={20} color={COLORS.navy} />
          <Text style={styles.linkText}>ScamAlert.sg</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.link}
          onPress={() => Linking.openURL("tel:18007226688")}
        >
          <Ionicons name="call-outline" size={20} color={COLORS.navy} />
          <Text style={styles.linkText}>Anti-Scam Hotline: 1800-722-6688</Text>
        </TouchableOpacity>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Disclaimer</Text>
        <Text style={styles.body}>
          This app is a research project by CS3264 (NUS). Results are indicative
          only and should not replace your own judgement. When in doubt, do not
          engage with the sender and report to the authorities.
        </Text>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: COLORS.offWhite },
  content: { paddingBottom: SPACING.xl },

  header: {
    backgroundColor: COLORS.navy,
    alignItems: "center",
    paddingVertical: SPACING.xl,
    paddingTop: 60,
  },
  headerTitle: { ...FONTS.title, color: COLORS.white, marginTop: SPACING.sm },
  headerSub: { ...FONTS.caption, color: COLORS.greyLight },

  section: { padding: SPACING.lg },
  sectionTitle: { ...FONTS.subtitle, marginBottom: SPACING.sm },
  body: { ...FONTS.regular, lineHeight: 24, color: COLORS.black },

  link: {
    flexDirection: "row",
    alignItems: "center",
    gap: SPACING.sm,
    paddingVertical: SPACING.sm,
  },
  linkText: { ...FONTS.regular, color: COLORS.navy, fontWeight: "500" },
});
