import React from "react";
import {
  View,
  Text,
  FlatList,
  StyleSheet,
  TouchableOpacity,
  Image,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";

import { COLORS, FONTS, SPACING, RISK } from "../constants/theme";
import useScanHistory from "../hooks/useScanHistory";

function timeAgo(isoString) {
  const diff = Date.now() - new Date(isoString).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins} min ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs} hr ago`;
  const days = Math.floor(hrs / 24);
  return `${days}d ago`;
}

function ScanCard({ item }) {
  const config = RISK[item.risk_level] || RISK.low;
  const pct = Math.round((item.risk_score ?? 0) * 100);

  return (
    <View style={styles.card}>
      {item.imageUri && (
        <Image source={{ uri: item.imageUri }} style={styles.thumb} />
      )}
      <View style={styles.cardBody}>
        <View style={styles.cardHeader}>
          <View
            style={[styles.badge, { backgroundColor: config.color }]}
          >
            <Text style={styles.badgeText}>
              {item.risk_level?.toUpperCase()}
            </Text>
          </View>
          <Text style={styles.pct}>{pct}% risk</Text>
        </View>
        <Text style={styles.time}>{timeAgo(item.timestamp)}</Text>
      </View>
    </View>
  );
}

export default function HistoryScreen() {
  const { history, clearHistory } = useScanHistory();

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>Scan History</Text>
        {history.length > 0 && (
          <TouchableOpacity onPress={clearHistory}>
            <Text style={styles.clearText}>Clear</Text>
          </TouchableOpacity>
        )}
      </View>

      {history.length === 0 ? (
        <View style={styles.empty}>
          <Ionicons name="time-outline" size={48} color={COLORS.greyLight} />
          <Text style={styles.emptyText}>No scans yet</Text>
        </View>
      ) : (
        <FlatList
          data={history}
          keyExtractor={(item) => item.id}
          renderItem={({ item }) => <ScanCard item={item} />}
          contentContainerStyle={{ padding: SPACING.md }}
        />
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: COLORS.offWhite },

  header: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    padding: SPACING.lg,
    paddingBottom: SPACING.sm,
    backgroundColor: COLORS.navy,
  },
  title: { ...FONTS.title, color: COLORS.white, fontSize: 22 },
  clearText: { color: COLORS.greyLight, fontSize: 14 },

  card: {
    flexDirection: "row",
    backgroundColor: COLORS.white,
    borderRadius: 12,
    overflow: "hidden",
    marginBottom: SPACING.sm,
  },
  thumb: { width: 70, height: 70 },
  cardBody: { flex: 1, padding: SPACING.sm, justifyContent: "center" },
  cardHeader: { flexDirection: "row", alignItems: "center", gap: SPACING.sm },
  badge: {
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 6,
  },
  badgeText: { color: COLORS.white, fontWeight: "700", fontSize: 11 },
  pct: { ...FONTS.caption },
  time: { ...FONTS.caption, marginTop: 2 },

  empty: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    gap: SPACING.sm,
  },
  emptyText: { ...FONTS.regular, color: COLORS.grey },
});
