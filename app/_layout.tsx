import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import HomeScreen from './(tabs)/index';
import Camera from './(tabs)/camera';
import FileView from './(tabs)/fileView';
import { useState } from 'react';

const Tab = createBottomTabNavigator();

export default function App() {
const [savedImages, setSavedImages] = useState<string[]>([]);
  return (
    <Tab.Navigator>
      <Tab.Screen name="Home" component={HomeScreen} />
      <Tab.Screen name="Camera">
        {() => <Camera setSavedImages={setSavedImages} />}
      </Tab.Screen>
      <Tab.Screen name="File View">
        {() => <FileView savedImages={savedImages} />}
      </Tab.Screen>
    </Tab.Navigator>
  );
}