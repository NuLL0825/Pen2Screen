import React, { useEffect, useState } from 'react';
import { View, Text, FlatList, Image, StyleSheet, TouchableOpacity, Alert, Button, Clipboard, Modal, ImageBackground } from 'react-native';
import * as FileSystem from 'expo-file-system';

const IMAGE_DIR = FileSystem.documentDirectory + 'images/';
const TEXT_DIR = FileSystem.documentDirectory + 'texts/';

interface FileViewProps {
  savedImages: { uri: string; text: string }[];
}

const FileView: React.FC<FileViewProps> = ({ savedImages }) => {
  const [images, setImages] = useState<{ uri: string; text: string }[]>([]);
  const [selectedImage, setSelectedImage] = useState<{ uri: string; text: string } | null>(null);
  const [modalVisible, setModalVisible] = useState(false);

  // Load images and their corresponding LaTeX text from the filesystem
  useEffect(() => {
    const loadImages = async () => {
      const imageFiles = await FileSystem.readDirectoryAsync(IMAGE_DIR);

      const imageData = await Promise.all(
        imageFiles.map(async (file) => {
          const imageUri = `${IMAGE_DIR}${file}`;
          const textUri = `${TEXT_DIR}${file.replace('.jpg', '.txt')}`;

          try {
            const text = await FileSystem.readAsStringAsync(textUri);
            return { uri: imageUri, text };
          } catch (error) {
            console.error('Error reading text file:', error);
            return { uri: imageUri, text: 'Text not available' };
          }
        })
      );
      setImages(imageData);
    };

    loadImages();
  }, [savedImages]);

  // Open modal with full-size image and LaTeX when image is clicked
  const openImage = (item: { uri: string; text: string }) => {
    setSelectedImage(item);
    setModalVisible(true);
  };

  // Delete the image and its corresponding LaTeX file
  const deleteFile = async () => {
    if (selectedImage) {
      try {
        await FileSystem.deleteAsync(selectedImage.uri);
        const textUri = selectedImage.uri.replace('.jpg', '.txt');
        // await FileSystem.deleteAsync(textUri);

        // Remove the deleted image from the list
        setImages(images.filter(image => image.uri !== selectedImage.uri));
        setModalVisible(false);
        alert('File deleted successfully!');
      } catch (error) {
        console.error('Error deleting file:', error);
        alert('Failed to delete file.');
      }
    }
  };

  // Copy LaTeX text to clipboard
  const copyToClipboard = async () => {
    if (selectedImage) {
      await Clipboard.setString(selectedImage.text);
      alert('LaTeX copied to clipboard!');
    }
  };

  const renderItem = ({ item }: { item: { uri: string; text: string } }) => (
    <TouchableOpacity onPress={() => openImage(item)} style={styles.item}>
      <Image source={{ uri: item.uri }} style={styles.image} />
    </TouchableOpacity>
  );

  return (
    <ImageBackground
    source={require('@/assets/images/background.png')} // Change to your image URL or local path
    style={styles.backgroundImage}
  >
    <View style={styles.container}>
      <Text style={styles.title}>Saved Images</Text>
      <FlatList
        data={images}
        keyExtractor={(item) => item.uri}
        renderItem={renderItem}
        numColumns={2}
        columnWrapperStyle={styles.columnWrapper}
      />

      {/* Modal for full-size image and LaTeX */}
      {selectedImage && (
        <Modal visible={modalVisible} animationType="slide" onRequestClose={() => setModalVisible(false)}>
          <View style={styles.modalContainer}>
            <Image source={{ uri: selectedImage.uri }} style={styles.fullImage} />
            <Text style={styles.latex}>{selectedImage.text}</Text>
            <View style={styles.modalActions}>
              <Button title="Delete" onPress={deleteFile} color="red" />
              <Button title="Copy LaTeX" onPress={copyToClipboard} />
              <Button title="Close" onPress={() => setModalVisible(false)} />
            </View>
          </View>
        </Modal>
      )}
    </View>
    </ImageBackground>
  );
};

const styles = StyleSheet.create({
  backgroundImage: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  container: {
    flex: 1,
    padding: 16,
    alignItems: 'center',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 16,
  },
  columnWrapper: {
    justifyContent: 'space-between',
  },
  item: {
    margin: 5,
  },
  image: {
    width: 150,
    height: 150,
    resizeMode: 'contain',
  },
  modalContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    padding: 20,
  },
  fullImage: {
    width: '100%',
    height: 300,
    resizeMode: 'contain',
    marginBottom: 20,
  },
  latex: {
    fontSize: 16,
    color: '#fff',
    marginBottom: 20,
    textAlign: 'center',
  },
  modalActions: {
    width: '100%',
    alignItems: 'center',
    justifyContent: 'space-evenly',
  },
});

export default FileView;
