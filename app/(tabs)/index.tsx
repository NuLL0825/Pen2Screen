import { ThemedView } from '@/components/ThemedView';
import { ThemedText } from '@/components/ThemedText';
import React from 'react';
import { StyleSheet, View, Image, TouchableOpacity, ImageBackground } from 'react-native';
import { useNavigation } from '@react-navigation/native';


export default function HomeScreen() {
  const navigation = useNavigation();

  const navigateToCamera = () => {
    navigation.navigate('Camera'); // ignore error
  };

  return (
  <ImageBackground
    source={require('@/assets/images/background.png')} // Change to your image URL or local path
    style={styles.backgroundImage}
  >
    <ThemedView style={styles.container}>
      <View style={styles.header}>
        <Image
          source={require('@/assets/images/pen2screen-logo-2.png')} // Replace with your logo or illustration
          style={styles.logo}
        />
        <ThemedText type="title" style={styles.title}>
          Welcome to Pen2Screen!
        </ThemedText>
        <ThemedText style={styles.subtitle}>
          Your handwriting recognition companion
        </ThemedText>
      </View>

      <View style={styles.stepsContainer}>
        <ThemedText type="subtitle" style={styles.stepsTitle}>
          How to Use:
        </ThemedText>
        <ThemedText style={styles.step}>
          1. Navigate to the <ThemedText style={{ color: 'black', fontWeight: 'bold' }}>Camera</ThemedText> tab to capture handwritten text.
        </ThemedText>
        <ThemedText style={styles.step}>
          2. View your captured images in the <ThemedText style={{ color: 'black', fontWeight: 'bold' }}>File View</ThemedText> tab.
        </ThemedText>
        <ThemedText style={styles.step}>
          3. Process the images with our Python model for recognition.
        </ThemedText>
      </View>

      <TouchableOpacity style={styles.button} onPress={navigateToCamera}>
        <ThemedText type="button" style={styles.buttonText}> 
          Get Started
        </ThemedText>
      </TouchableOpacity> 
    </ThemedView> 
    
  </ImageBackground>
  );
}

const styles = StyleSheet.create({
  backgroundImage: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  container: {
    flex: 1,
    padding: 20,
    justifyContent: 'space-between',
    backgroundColor: 'rgba(249, 249, 249, 0.1)',
  },
  header: {
    alignItems: 'center',
    marginBottom: 20,
  },
  logo: {
    width: 100,
    height: 100,
    marginBottom: 10,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333', // Neutral text color
    textAlign: 'center',
  },
  subtitle: {
    fontSize: 16,
    color: '#666', // Subtle text color
    textAlign: 'center',
    marginTop: 5,
  },
  stepsContainer: {
    paddingVertical: 20,
  },
  stepsTitle: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 10,
    color: '#444',
  },
  step: {
    fontSize: 16,
    color: '#555',
    marginBottom: 8,
    lineHeight: 22,
  },
  button: {
    backgroundColor: '#4CAF50', // Calm green
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 8,
    alignItems: 'center',
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  highlight: { 
    fontSize: 16, 
    color: '#4CAF50', 
    fontWeight: 'bold' 
  },
});
