import { CameraView, useCameraPermissions } from 'expo-camera';
import { useState, useRef, useEffect } from 'react';
import { Button, StyleSheet, Text, View, Image, TouchableOpacity } from 'react-native';
import * as FileSystem from 'expo-file-system';
import * as ImagePicker from 'expo-image-picker';
import UUID from 'react-native-uuid';

const IMAGE_DIR = FileSystem.documentDirectory + 'images/';
console.log(IMAGE_DIR);

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

  const processImage = async (uri: string) => {
    const formData = new FormData();
    formData.append('image', {
        uri,
        name: 'image.jpg',
        type: 'image/jpeg',
    } as any);

    try {
        const response = await fetch('http://192.168.26.83:5000/process-image', {
            method: 'POST',
            body: formData,
        });

        const result = await response.json();
        if (response.ok) {
          const uniqueId = UUID.v4();
          const processedUri = `${FileSystem.documentDirectory}images/${uniqueId}.jpg`;
          await FileSystem.downloadAsync(
            result.processed_image_path, // Correct URL
            processedUri
          );

          setSavedImages((prev) => [...prev, processedUri]); // ignore?? (gumagana amp)
          setImageUri(processedUri);
          alert('Image processed successfully!');
        } else {
            alert(`Error: ${result.error}`);
        }
    } catch (error) {
        console.error(error);
        alert('Failed to process image.');
    }
};

  const captureImage = async () => {
    if (cameraRef.current) {
      const photo = await cameraRef.current.takePictureAsync();
      if (photo && photo.uri) {
        await processImage(photo.uri); // Process the captured image
      }
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
      await processImage(result.assets[0].uri); // Process the picked image
    }
  };

  if (permission === null) {
    return (
      <View style={styles.container}>
        <Text>Loading camera permissions...</Text>
      </View>
    );
  }

  if (permission.status === 'denied') {
    return (
      <View style={styles.container}>
        <Text style={styles.errorText}>Camera permission is required. Please grant permission.</Text>
        <Button title="Request Permission" onPress={requestPermission} />
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <CameraView style={styles.camera} ref={cameraRef}>
        <View style={styles.buttonContainer}>
          <TouchableOpacity style={styles.button} onPress={captureImage}>
            <Text style={styles.buttonText}>Capture</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.button} onPress={pickImage}>
            <Text style={styles.buttonText}>Pick an Image</Text>
          </TouchableOpacity>
        </View>
      </CameraView>
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
  button: {
    backgroundColor: '#4CAF50',
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 16,
    alignItems: 'center',
    width: '40%',
    marginVertical: 8,
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
    textAlign: 'center',
  },
});
