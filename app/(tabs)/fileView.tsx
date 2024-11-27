import React, { useEffect, useState } from 'react';
import { View, Text, FlatList, Image, StyleSheet } from 'react-native';
import * as FileSystem from 'expo-file-system';

const IMAGE_DIR = FileSystem.documentDirectory + 'images/';

interface FileViewProps {
  savedImages: string[];
}

const FileView: React.FC<FileViewProps> = ({ savedImages }) => {
  const [images, setImages] = useState<string[]>([]);

  useEffect(() => {
    const loadImages = async () => {
      const files = await FileSystem.readDirectoryAsync(IMAGE_DIR);
      const imagePaths = files.map((file) => `${IMAGE_DIR}${file}`);
      setImages(imagePaths);
    };

    loadImages();
  }, [savedImages]); // Reload images whenever savedImages changes

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Saved Images</Text>
      <FlatList
        data={images}
        keyExtractor={(item) => item}
        renderItem={({ item }) => <Image source={{ uri: item }} style={styles.image} />}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 16,
  },
  title: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 16,
  },
  image: {
    width: 100,
    height: 100,
    marginBottom: 8,
  },
});

export default FileView;
