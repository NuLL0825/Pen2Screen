import React, { useEffect, useState } from 'react';
import { View, Text, FlatList, Image, StyleSheet, Dimensions } from 'react-native';
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
  }, [savedImages]);

  const renderItem = ({ item }: { item: string }) => (
    <Image source={{ uri: item }} style={styles.image} />
  );

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Saved Images</Text>
      <FlatList
        data={images}
        keyExtractor={(item) => item}
        renderItem={renderItem}
        numColumns={2} // Display images in two columns
        columnWrapperStyle={styles.columnWrapper} // Style for the column wrapper
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 16,
    alignItems: 'center', // Center items horizontally
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 16,
  },
  columnWrapper: {
    justifyContent: 'space-between', // Space out the columns
  },
  image: {
    width: (Dimensions.get('window').width / 2) - 20, // Make images larger and responsive
    height: 200, // Set a fixed height for images
    borderRadius: 10,
    marginBottom: 2,
    marginLeft: 1,
    marginRight: 1,
  },
});

export default FileView;