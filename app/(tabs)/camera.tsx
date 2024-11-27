import { CameraView, useCameraPermissions } from 'expo-camera';
import { useState, useRef, useEffect } from 'react';
import { Button, StyleSheet, Text, View, Image, TouchableOpacity } from 'react-native';
import * as FileSystem from 'expo-file-system';
import * as ImagePicker from 'expo-image-picker';

const IMAGE_DIR = FileSystem.documentDirectory + 'images/';

interface AppProps {
  setSavedImages: (images: string[]) => void;
}

export default function Camera({ setSavedImages }: AppProps) {
  const [permission, requestPermission] = useCameraPermissions();
  const [imageUri, setImageUri] = useState<string | null>(null);
  const cameraRef = useRef<CameraView>(null);

  useEffect(() => {
    const createDirectory = async () => {
      await FileSystem.makeDirectoryAsync(IMAGE_DIR, { intermediates: true });
    };
    createDirectory();
  }, []);

  const captureImage = async () => {
    if (cameraRef.current) {
      const photo = await cameraRef.current.takePictureAsync();
      if (photo && photo.uri) {
        setImageUri(photo.uri);
      }
    }
  };

  const saveImage = async (uri: string) => {
    const fileName = uri.split('/').pop();
    const newPath = `${IMAGE_DIR}${fileName}`;
    await FileSystem.moveAsync({ from: uri, to: newPath });
    setSavedImages((prev) => [...prev, newPath]);
  };

  const handleImagePress = async () => {
    if (imageUri) {
      await saveImage(imageUri);
      alert('Image saved successfully!');
    }
  };

  const pickImage = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      allowsEditing: true,
      aspect: [4, 3],
      quality: 1,
    });

    if (!result.canceled) {
      setImageUri(result.assets[0].uri);
    }
  };

  // Check if permission is granted
  if (permission === null) {
    return (
      <View style={styles.container}>
        <Text>Loading camera permissions...</Text>
      </View>
    );
  }

  // Show error message if permission is denied
  if (permission.status === 'denied') {
    return (
      <View style={styles.container}>
        <Text style={styles.errorText}>
          Camera permission is required. Please grant permission.
        </Text>
        <Button title="Request Permission" onPress={requestPermission} />
      </View>
    );
  }

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
    height: 300,
    borderRadius: 10,
    marginTop: 10,
  },
  errorText: {
    color: 'red',
    fontSize: 16,
    textAlign: 'center',
    marginBottom: 10,
  },
});
