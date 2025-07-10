import React, { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { SimulationState } from '../types/simulation';

interface LastGeneratedImageProps {
  state: SimulationState;
}

export const LastGeneratedImage: React.FC<LastGeneratedImageProps> = ({ state }) => {
  const [imageData, setImageData] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showModal, setShowModal] = useState(false);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [totalImages, setTotalImages] = useState(0);

  const loadImageCount = async () => {
    try {
      const count = await invoke<number>('get_narrative_images_count');
      setTotalImages(count);
    } catch (err) {
      console.error('Failed to load image count:', err);
    }
  };

  const loadImage = async (index: number = 0) => {
    setLoading(true);
    setError(null);
    try {
      const dataUrl = await invoke<string>('get_narrative_image_by_index', { index });
      setImageData(dataUrl);
      setCurrentIndex(index);
    } catch (err) {
      setError(err as string);
      console.error('Failed to load image:', err);
    } finally {
      setLoading(false);
    }
  };

  const nextImage = () => {
    if (currentIndex < totalImages - 1) {
      loadImage(currentIndex + 1);
    }
  };

  const prevImage = () => {
    if (currentIndex > 0) {
      loadImage(currentIndex - 1);
    }
  };

  useEffect(() => {
    loadImageCount();
    loadImage(0);
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-48 bg-gray-50 rounded-lg border-2 border-dashed border-gray-300">
        <div className="text-center text-gray-500">
          <div className="text-2xl mb-2">⏳</div>
          <p className="text-sm font-medium">Loading image...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-48 bg-gray-50 rounded-lg border-2 border-dashed border-gray-300">
        <div className="text-center text-gray-500">
          <div className="text-2xl mb-2">❌</div>
          <p className="text-sm font-medium">Failed to load image</p>
          <p className="text-xs mt-1 text-gray-400">{error}</p>
          <button
            onClick={() => loadImage(currentIndex)}
            className="mt-2 px-3 py-1 bg-blue-500 text-white text-xs rounded hover:bg-blue-600"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!imageData) {
    return (
      <div className="flex items-center justify-center h-48 bg-gray-50 rounded-lg border-2 border-dashed border-gray-300">
        <div className="text-center text-gray-500">
          <div className="text-4xl mb-2">🖼️</div>
          <p className="text-sm font-medium">No image available</p>
          <button
            onClick={() => loadImage(0)}
            className="mt-2 px-3 py-1 bg-blue-500 text-white text-xs rounded hover:bg-blue-600"
          >
            Load Image
          </button>
        </div>
      </div>
    );
  }

  return (
    <>
      <div className="bg-gray-50 rounded-lg border border-gray-300 overflow-hidden relative">
        <img
          src={imageData}
          alt="Latest Generated Narrative"
          className="w-full h-auto max-h-96 object-contain cursor-pointer hover:opacity-90 transition-opacity"
          onClick={() => setShowModal(true)}
        />
        
        {/* Navigation arrows */}
        {totalImages > 1 && (
          <>
            <button
              onClick={prevImage}
              disabled={currentIndex === 0}
              className={`absolute left-2 top-1/2 transform -translate-y-1/2 bg-black bg-opacity-50 text-white rounded-full w-8 h-8 flex items-center justify-center hover:bg-opacity-75 transition-opacity ${
                currentIndex === 0 ? 'opacity-30 cursor-not-allowed' : 'opacity-70 hover:opacity-100'
              }`}
            >
              ←
            </button>
            <button
              onClick={nextImage}
              disabled={currentIndex === totalImages - 1}
              className={`absolute right-2 top-1/2 transform -translate-y-1/2 bg-black bg-opacity-50 text-white rounded-full w-8 h-8 flex items-center justify-center hover:bg-opacity-75 transition-opacity ${
                currentIndex === totalImages - 1 ? 'opacity-30 cursor-not-allowed' : 'opacity-70 hover:opacity-100'
              }`}
            >
              →
            </button>
          </>
        )}
        
        {/* Image counter */}
        {totalImages > 1 && (
          <div className="absolute bottom-2 left-1/2 transform -translate-x-1/2 bg-black bg-opacity-50 text-white px-2 py-1 rounded text-xs">
            {currentIndex + 1} / {totalImages}
          </div>
        )}
      </div>

      {/* Modal for larger image view */}
      {showModal && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 p-4"
          onClick={() => setShowModal(false)}
        >
          <div className="relative max-w-4xl max-h-full">
            <button
              onClick={() => setShowModal(false)}
              className="absolute top-4 right-4 text-white bg-black bg-opacity-50 rounded-full w-8 h-8 flex items-center justify-center hover:bg-opacity-75 z-10"
            >
              ×
            </button>
            
            {/* Navigation arrows in modal */}
            {totalImages > 1 && (
              <>
                <button
                  onClick={(e) => { e.stopPropagation(); prevImage(); }}
                  disabled={currentIndex === 0}
                  className={`absolute left-4 top-1/2 transform -translate-y-1/2 bg-black bg-opacity-50 text-white rounded-full w-12 h-12 flex items-center justify-center hover:bg-opacity-75 transition-opacity text-xl z-10 ${
                    currentIndex === 0 ? 'opacity-30 cursor-not-allowed' : 'opacity-70 hover:opacity-100'
                  }`}
                >
                  ←
                </button>
                <button
                  onClick={(e) => { e.stopPropagation(); nextImage(); }}
                  disabled={currentIndex === totalImages - 1}
                  className={`absolute right-4 top-1/2 transform -translate-y-1/2 bg-black bg-opacity-50 text-white rounded-full w-12 h-12 flex items-center justify-center hover:bg-opacity-75 transition-opacity text-xl z-10 ${
                    currentIndex === totalImages - 1 ? 'opacity-30 cursor-not-allowed' : 'opacity-70 hover:opacity-100'
                  }`}
                >
                  →
                </button>
              </>
            )}
            
            {/* Image counter in modal */}
            {totalImages > 1 && (
              <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 bg-black bg-opacity-50 text-white px-3 py-2 rounded z-10">
                {currentIndex + 1} / {totalImages}
              </div>
            )}
            
            <img
              src={imageData}
              alt="Latest Generated Narrative (Full Size)"
              className="max-w-full max-h-full object-contain rounded-lg"
              onClick={(e) => e.stopPropagation()}
            />
          </div>
        </div>
      )}
    </>
  );
};