import { ThemedView } from '@/components/ThemedView';
import { ThemedText } from '@/components/ThemedText';
import React from 'react';

export default function HomeScreen() {
  return (
    <ThemedView style={{ flex: 1, padding: 20 }}>
      <ThemedText type="title">Welcome to Pen2Screen!</ThemedText>
      <ThemedText>Follow these steps to use the app:</ThemedText>
      <ThemedText>1. Navigate to the Camera tab to capture handwritten text.</ThemedText>
      <ThemedText>2. Use the File View tab to see your captured images.</ThemedText>
      <ThemedText>3. Process the images with your Python model.</ThemedText>
    </ThemedView>
  );
}