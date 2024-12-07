import React, { useEffect, useState } from 'react';
import { View, Text, FlatList, Image, StyleSheet, TouchableOpacity } from 'react-native';
import * as FileSystem from 'expo-file-system';

const IMAGE_DIR = FileSystem.documentDirectory + 'images/';
const TEXT_DIR = FileSystem.documentDirectory + 'texts/';

interface FileViewProps {
  savedImages: { uri: string; text: string }[];
}

const FileView: React.FC<FileViewProps> = ({ savedImages}) => {
  const [images, setImages] = useState<{ uri: string; text: string }[]>([]);

  useEffect(() => {
    const loadImages = async () => {
      const imageFiles = await FileSystem.readDirectoryAsync(IMAGE_DIR);
      // console.log('Loaded image files:', imageFiles);

      const imageData = await Promise.all(
        imageFiles.map(async (file) => {
          const imageUri = `${IMAGE_DIR}${file}`;
          const textUri = `${TEXT_DIR}${file.replace('.jpg', '.txt')}`;
          // console.log('Text URI:', textUri); 

          try {
            const text = await FileSystem.readAsStringAsync(textUri);
            return { uri: imageUri, text };
          } catch (error) {
            console.error('Error reading text file:', error);
            return { uri: imageUri, text: 'Text not available' };
          }
        })
      );
      // console.log('Loaded image data:', imageData);
      setImages(imageData);
    };
    
    loadImages();
  }, [savedImages]); // constanly rendering

  const renderItem = ({ item }: { item: { uri: string; text: string } }) => (
    <TouchableOpacity onPress={() => alert(item.text)}>
      <Image source={{ uri: item.uri }} style={styles.image} />
    </TouchableOpacity>
  );

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Saved Images</Text>
      <FlatList
        key={savedImages.length.toString()} 
        data={images}
        keyExtractor={(item) => item.uri}
        renderItem={renderItem}
        numColumns={2}
        columnWrapperStyle={styles.columnWrapper}
      />
    </View>
  );
};

const styles = StyleSheet.create({
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
  image: {
    width: 150,
    height: 150,
    margin: 5,
  },
});

export default FileView;
