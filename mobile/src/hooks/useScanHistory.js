import { useCallback, useEffect, useState } from "react";
import AsyncStorage from "@react-native-async-storage/async-storage";

const STORAGE_KEY = "@scamcheck_history";
const MAX_ITEMS = 50;

/**
 * Persisted scan history stored in AsyncStorage.
 *
 * Each entry: { id, imageUri, risk_level, risk_score, reasons, timestamp }
 */
export default function useScanHistory() {
  const [history, setHistory] = useState([]);

  useEffect(() => {
    AsyncStorage.getItem(STORAGE_KEY).then((raw) => {
      if (raw) setHistory(JSON.parse(raw));
    });
  }, []);

  const persist = useCallback(async (next) => {
    setHistory(next);
    await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  }, []);

  const addScan = useCallback(
    async (entry) => {
      const item = {
        id: Date.now().toString(),
        timestamp: new Date().toISOString(),
        ...entry,
      };
      const next = [item, ...history].slice(0, MAX_ITEMS);
      await persist(next);
      return item;
    },
    [history, persist]
  );

  const clearHistory = useCallback(async () => {
    await persist([]);
  }, [persist]);

  return { history, addScan, clearHistory };
}
