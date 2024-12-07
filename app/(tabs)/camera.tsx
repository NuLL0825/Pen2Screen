import { CameraView, useCameraPermissions } from 'expo-camera';
import { useState, useRef, useEffect } from 'react';
import { Button, StyleSheet, Text, View, TouchableOpacity, ActivityIndicator } from 'react-native';
import * as FileSystem from 'expo-file-system';
import * as ImagePicker from 'expo-image-picker';

const IMAGE_DIR = FileSystem.documentDirectory + 'images/';
const TEXT_DIR = FileSystem.documentDirectory + 'texts/';

interface CameraProps {
  loadImages: () => void;
}

export default function Camera({ loadImages }: CameraProps) {
  const [permission, requestPermission] = useCameraPermissions();
  const [imageUri, setImageUri] = useState<string | null>(null);
  const cameraRef = useRef<CameraView>(null);
  const [loading, setLoading] = useState(false);


  useEffect(() => {
    const createDirectories = async () => {
      await FileSystem.makeDirectoryAsync(IMAGE_DIR, { intermediates: true });
      await FileSystem.makeDirectoryAsync(TEXT_DIR, { intermediates: true });
    };
    createDirectories();
  }, []);

  const processImage = async (uri: string) => {
    setLoading(true);
    const formData = new FormData();
    formData.append('image', {
      uri,
      name: 'image.jpg',
      type: 'image/jpeg',
    });

    try {
      const response = await fetch('http://192.168.26.83:5000/process-image', {
        method: 'POST',
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        body: formData,
      });

      const result = await response.json();
      if (response.ok) {
        const imageName = uri.split('/').pop();
        const text = result.text || 'No text extracted';

        // Save image and extracted text
        const newImageUri = IMAGE_DIR + imageName;
        const newTextUri = TEXT_DIR + imageName.replace('.jpg', '.txt');

        await FileSystem.copyAsync({ from: uri, to: newImageUri });
        await FileSystem.writeAsStringAsync(newTextUri, text);

        alert('Image and text saved locally!');
      } else {
        alert(`Error: ${result.error}`);
      }
    } catch (error) {
      console.error(error);
      alert('Failed to process image.');
    } finally {
      setLoading(false); // End loading
    }
  };

  const captureImage = async () => {
    if (cameraRef.current) {
      const photo = await cameraRef.current.takePictureAsync();
      if (photo && photo.uri) {
        await processImage(photo.uri);
        loadImages();
      }
    }
  };

  const pickImage = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      allowsEditing: true,
      quality: 1,
    });

    if (!result.canceled) {
      await processImage(result.assets[0].uri);
    }
  };

  if (permission === null) {
    return <Text>Loading camera permissions...</Text>;
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
      {loading ? (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#4CAF50" />
          <Text style={styles.loadingText}>Processing image, please wait...</Text>
        </View>
      ) : (
        <CameraView style={styles.camera} ref={cameraRef}>
          <TouchableOpacity style={styles.button} onPress={captureImage}>
            <Text style={styles.buttonText}>Capture</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.button} onPress={pickImage}>
            <Text style={styles.buttonText}>Pick an Image</Text>
          </TouchableOpacity>
        </CameraView>
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
  button: {
    backgroundColor: '#4CAF50',
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 16,
    marginBottom: 10,
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
  },
  errorText: {
    color: 'red',
    fontSize: 16,
    textAlign: 'center',
    marginBottom: 10,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 10,
    fontSize: 16,
    color: '#4CAF50',
  },
});
