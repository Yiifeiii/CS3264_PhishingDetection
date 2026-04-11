import React from "react";
import { StatusBar } from "expo-status-bar";
import { NavigationContainer } from "@react-navigation/native";
import { createBottomTabNavigator } from "@react-navigation/bottom-tabs";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import { Ionicons } from "@expo/vector-icons";

import HomeScreen from "./src/screens/HomeScreen";
import PreviewScreen from "./src/screens/PreviewScreen";
import AnalysingScreen from "./src/screens/AnalysingScreen";
import ResultScreen from "./src/screens/ResultScreen";
import HistoryScreen from "./src/screens/HistoryScreen";
import AboutScreen from "./src/screens/AboutScreen";

import { COLORS } from "./src/constants/theme";

const Stack = createNativeStackNavigator();
const Tab = createBottomTabNavigator();

function ScanStack() {
  return (
    <Stack.Navigator
      screenOptions={{
        headerStyle: { backgroundColor: COLORS.navy },
        headerTintColor: COLORS.white,
        headerTitleStyle: { fontWeight: "600" },
      }}
    >
      <Stack.Screen
        name="Home"
        component={HomeScreen}
        options={{ title: "ScamCheck SG" }}
      />
      <Stack.Screen
        name="Preview"
        component={PreviewScreen}
        options={{ title: "Review Image" }}
      />
      <Stack.Screen
        name="Analysing"
        component={AnalysingScreen}
        options={{ title: "Analysing...", headerBackVisible: false }}
      />
      <Stack.Screen
        name="Result"
        component={ResultScreen}
        options={{ title: "Result" }}
      />
    </Stack.Navigator>
  );
}

const TAB_ICONS = {
  Scan: { focused: "scan-circle", unfocused: "scan-circle-outline" },
  History: { focused: "time", unfocused: "time-outline" },
  About: {
    focused: "information-circle",
    unfocused: "information-circle-outline",
  },
};

export default function App() {
  return (
    <NavigationContainer>
      <StatusBar style="light" />
      <Tab.Navigator
        screenOptions={({ route }) => ({
          headerShown: false,
          tabBarActiveTintColor: COLORS.navy,
          tabBarInactiveTintColor: COLORS.grey,
          tabBarIcon: ({ focused, color, size }) => {
            const icons = TAB_ICONS[route.name];
            const iconName = focused ? icons.focused : icons.unfocused;
            return <Ionicons name={iconName} size={size} color={color} />;
          },
        })}
      >
        <Tab.Screen name="Scan" component={ScanStack} />
        <Tab.Screen name="History" component={HistoryScreen} />
        <Tab.Screen name="About" component={AboutScreen} />
      </Tab.Navigator>
    </NavigationContainer>
  );
}
