import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import HomeScreen from './(tabs)/index';
import Camera from './(tabs)/camera';
import FileView from './(tabs)/fileView';
import { useState } from 'react';

import Entypo from '@expo/vector-icons/Entypo';
import Ionicons from '@expo/vector-icons/Ionicons';

const Tab = createBottomTabNavigator();

export default function App() {
const [savedImages, setSavedImages] = useState<string[]>([]);
  return (
    <Tab.Navigator>
      <Tab.Screen name="Home" component={HomeScreen} options={{
            headerShown: false,
            tabBarIcon: ({ color, size }) => (
              <Entypo name="home" size={24} color="black" />
            ),
          }}/>
      <Tab.Screen name="Camera" options={{
            headerShown: false,
            tabBarIcon: ({ color, size }) => (
              <Ionicons name="scan-circle-sharp" size={24} color="black" />
            ),
          }}>
        {() => <Camera setSavedImages={setSavedImages} />}
      </Tab.Screen>
      <Tab.Screen name="File View" options={{
            headerShown: false,
            tabBarIcon: ({ color, size }) => (
              <Entypo name="folder-images" size={24} color="black" />
            ),
          }}>
        {() => <FileView savedImages={savedImages} />}
      </Tab.Screen>
    </Tab.Navigator>
  );
}