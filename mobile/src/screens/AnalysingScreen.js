import React, { useEffect, useRef, useState } from "react";
import { View, Text, StyleSheet, Animated } from "react-native";
import { Ionicons } from "@expo/vector-icons";

import { COLORS, FONTS, SPACING } from "../constants/theme";
import { analyseImage } from "../services/api";

const STEPS = [
  "Extracting text from image...",
  "Scanning visual cues...",
  "Running AI model...",
];

export default function AnalysingScreen({ route, navigation }) {
  const { imageUri } = route.params;
  const [stepIndex, setStepIndex] = useState(0);
  const pulseAnim = useRef(new Animated.Value(1)).current;

  // Pulsing shield animation
  useEffect(() => {
    const loop = Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, {
          toValue: 1.15,
          duration: 600,
          useNativeDriver: true,
        }),
        Animated.timing(pulseAnim, {
          toValue: 1,
          duration: 600,
          useNativeDriver: true,
        }),
      ])
    );
    loop.start();
    return () => loop.stop();
  }, [pulseAnim]);

  // Simulate step progression while waiting for the API.
  useEffect(() => {
    const timers = STEPS.map((_, i) =>
      setTimeout(() => setStepIndex(i), i * 2000)
    );
    return () => timers.forEach(clearTimeout);
  }, []);

  // Fire the actual analysis request.
  useEffect(() => {
    let cancelled = false;

    analyseImage(imageUri)
      .then((result) => {
        if (!cancelled) {
          navigation.replace("Result", { imageUri, result });
        }
      })
      .catch((err) => {
        if (!cancelled) {
          navigation.replace("Result", {
            imageUri,
            result: {
              risk_level: "low",
              risk_score: 0,
              image_score: 0,
              text_score: 0,
              reasons: [`Analysis failed: ${err.message}`],
            },
          });
        }
      });

    return () => {
      cancelled = true;
    };
  }, [imageUri, navigation]);

  return (
    <View style={styles.container}>
      <Animated.View style={{ transform: [{ scale: pulseAnim }] }}>
        <Ionicons
          name="shield-checkmark"
          size={72}
          color={COLORS.navy}
        />
      </Animated.View>

      <Text style={styles.title}>Analysing your image...</Text>

      <View style={styles.steps}>
        {STEPS.map((step, i) => {
          const done = i < stepIndex;
          const active = i === stepIndex;
          return (
            <View key={step} style={styles.stepRow}>
              <Ionicons
                name={done ? "checkmark-circle" : active ? "time" : "ellipse-outline"}
                size={20}
                color={done ? COLORS.green : active ? COLORS.amber : COLORS.greyLight}
              />
              <Text
                style={[
                  styles.stepText,
                  done && { color: COLORS.green },
                  active && { color: COLORS.black, fontWeight: "600" },
                ]}
              >
                {step}
              </Text>
            </View>
          );
        })}
      </View>

      <Text style={styles.estimate}>This usually takes 5-10 seconds.</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.offWhite,
    alignItems: "center",
    justifyContent: "center",
    padding: SPACING.xl,
  },
  title: { ...FONTS.subtitle, marginTop: SPACING.lg, marginBottom: SPACING.xl },

  steps: { width: "100%", gap: SPACING.md },
  stepRow: { flexDirection: "row", alignItems: "center", gap: SPACING.sm },
  stepText: { ...FONTS.regular, color: COLORS.greyLight },

  estimate: { ...FONTS.caption, marginTop: SPACING.xl },
});
