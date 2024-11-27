import { CameraView, useCameraPermissions } from 'expo-camera';
import { useState, useRef, useEffect } from 'react';
import { Button, StyleSheet, Text, View, Image, TouchableOpacity } from 'react-native';
import * as FileSystem from 'expo-file-system';
import * as ImagePicker from 'expo-image-picker';

const IMAGE_DIR = FileSystem.documentDirectory + 'images/'; // Directory to save images

interface AppProps {
  setSavedImages: (images: string[]) => void; // Ensure this is typed correctly
}

export default function Camera({ setSavedImages }: AppProps) {
  const [permission, requestPermission] = useCameraPermissions();
  const [imageUri, setImageUri] = useState<string | null>(null);
  const cameraRef = useRef<CameraView>(null);

  useEffect(() => {
    // Ensure the image directory exists
    const createDirectory = async () => {
      await FileSystem.makeDirectoryAsync(IMAGE_DIR, { intermediates: true });
    };
    createDirectory();
  }, []);

  const captureImage = async () => {
    if (cameraRef.current) {
      const photo = await cameraRef.current.takePictureAsync();
      if (photo && photo.uri) {
        setImageUri(photo.uri); // Set the captured image URI
      }
    }
  };

  const saveImage = async (uri: string) => {
    const fileName = uri.split('/').pop(); // Get the file name from the URI
    const newPath = `${IMAGE_DIR}${fileName}`; // Define the new path

    // Move the file to the new path
    await FileSystem.moveAsync({
      from: uri,
      to: newPath,
    });

    console.log(`Image saved to: ${newPath}`);
    setSavedImages((prev: string[]) => [...prev, newPath]) // Ignore (works fine anyways)
  };

  const handleImagePress = async () => {
    if (imageUri) {
      await saveImage(imageUri); // Save the image when pressed
      alert('Image saved successfully!'); // Optional: Show a success message
    }
  };

  const pickImage = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ["images"],
      allowsEditing: true,
      aspect: [4, 3],
      quality: 1,
    });

    if (!result.canceled) {
      setImageUri(result.assets[0].uri); // Set the picked image URI
    }
  };

  return (
    <View style={styles.container}>
      <CameraView style={styles.camera} ref={cameraRef}>
        <View style={styles.buttonContainer}>
          <Button title="Capture" onPress={captureImage} />
          <Button title="Pick an Image" onPress={pickImage} />
        </View>
      </CameraView>
      {imageUri && (
        <TouchableOpacity onPress={handleImagePress} style={styles.imageContainer}>
          <Text style={styles.imageText}>Captured Image:</Text>
          <Image source={{ uri: imageUri }} style={styles.capturedImage} />
        </TouchableOpacity>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
  },
  camera: {
    flex: 1,
  },
  buttonContainer: {
    position: 'absolute',
    bottom: 20,
    left: 0,
    right: 0,
    alignItems: 'center',
  },
  imageContainer: {
    alignItems: 'center',
    marginTop: 20,
  },
  imageText: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  capturedImage: {
    width: 300,
    height: 300 , 
    borderRadius: 10,
    marginTop: 10,
  },
});