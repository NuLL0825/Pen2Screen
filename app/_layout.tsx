import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import HomeScreen from './(tabs)/index';
import Camera from './(tabs)/camera';
import FileView from './(tabs)/fileView';
import { useState, useEffect } from 'react';
import * as FileSystem from 'expo-file-system';

import Entypo from '@expo/vector-icons/Entypo';
import Ionicons from '@expo/vector-icons/Ionicons';

const Tab = createBottomTabNavigator();

const IMAGE_DIR = FileSystem.documentDirectory + 'images/';
const TEXT_DIR = FileSystem.documentDirectory + 'texts/';

export default function App() {
  const [savedImages, setSavedImages] = useState<{ uri: string; text: string }[]>([]);

  // Load images when the app starts
  useEffect(() => {
    const loadImages = async () => {
      try {
        const imageFiles = await FileSystem.readDirectoryAsync(IMAGE_DIR);
        const imageData = await Promise.all(
          imageFiles.map(async (file) => {
            const imageUri = `${IMAGE_DIR}${file}`;
            const textUri = `${TEXT_DIR}${file.replace('.jpg', '.txt')}`;
            try {
              const text = await FileSystem.readAsStringAsync(textUri);
              return { uri: imageUri, text };
            } catch {
              return { uri: imageUri, text: 'Text not available' };
            }
          })
        );
        setSavedImages(imageData); // Update saved images
      } catch (error) {
        console.error('Error loading images:', error);
      }
    };

    loadImages();
  }, [savedImages]); // Run once when the app starts

  return (
    <Tab.Navigator>
      <Tab.Screen
        name="Home"
        component={HomeScreen}
        options={{
          headerShown: false,
          tabBarIcon: ({ color, size }) => <Entypo name="home" size={24} color="black" />,
        }}
      />
      <Tab.Screen
        name="Camera"
        options={{
          headerShown: false,
          tabBarIcon: ({ color, size }) => <Ionicons name="scan-circle-sharp" size={24} color="black" />,
        }}
      >
        {() => <Camera loadImages={savedImages} />}
      </Tab.Screen>
      <Tab.Screen
        name="File View"
        options={{
          headerShown: false,
          tabBarIcon: ({ color, size }) => <Entypo name="folder-images" size={24} color="black" />,
        }}
      >
        {() => <FileView savedImages={savedImages} />}
      </Tab.Screen>
    </Tab.Navigator>
  );
}
