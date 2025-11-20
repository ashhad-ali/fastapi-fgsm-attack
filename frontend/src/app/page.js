'use client';
import { useState } from 'react';

export default function Home() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [epsilon, setEpsilon] = useState(0.1);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setResult(null); // Reset results when new image is picked
      setError(null);
    }
  };

  const handleAttack = async () => {
    if (!file) {
      alert("Please select an image first!");
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('epsilon', epsilon);

    try {
      // We use 127.0.0.1 instead of localhost to avoid common network glitches
      const response = await fetch('http://51.20.12.114/attack', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error(err);
      setError("Failed to connect to the backend. Is the server running?");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-gray-900 text-white p-8 font-sans">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-4xl font-bold mb-2 text-blue-400">DevNeuron AI Assessment</h1>
        <p className="text-gray-400 mb-8">Adversarial Attack Demonstration (FGSM)</p>

        {/* Controls Section */}
        <div className="bg-gray-800 p-6 rounded-lg shadow-lg mb-8 border border-gray-700">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            
            {/* Upload Input */}
            <div>
              <label className="block text-sm font-medium mb-2 text-gray-300">1. Upload Target Image (Digit)</label>
              <input 
                type="file" 
                accept="image/*"
                onChange={handleFileChange}
                className="block w-full text-sm text-gray-400
                  file:mr-4 file:py-2 file:px-4
                  file:rounded-full file:border-0
                  file:text-sm file:font-semibold
                  file:bg-blue-600 file:text-white
                  hover:file:bg-blue-700
                  cursor-pointer bg-gray-700 rounded-lg"
              />
            </div>

            {/* Epsilon Slider */}
            <div>
              <label className="block text-sm font-medium mb-2 text-gray-300">
                2. Set Attack Strength (Epsilon): <span className="text-blue-400 font-bold">{epsilon}</span>
              </label>
              <input 
                type="range" 
                min="0" 
                max="0.5" 
                step="0.01" 
                value={epsilon} 
                onChange={(e) => setEpsilon(parseFloat(e.target.value))}
                className="w-full h-2 bg-gray-600 rounded-lg appearance-none cursor-pointer accent-blue-500"
              />
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>0 (No Attack)</span>
                <span>0.5 (Heavy Noise)</span>
              </div>
            </div>
          </div>

          {/* Run Button */}
          <div className="mt-6 text-center">
            <button 
              onClick={handleAttack} 
              disabled={loading || !file}
              className={`px-8 py-3 rounded-full font-bold text-lg transition-all ${
                loading || !file 
                  ? 'bg-gray-600 cursor-not-allowed opacity-50' 
                  : 'bg-blue-600 hover:bg-blue-500 hover:scale-105 shadow-blue-500/50 shadow-lg'
              }`}
            >
              {loading ? 'Running Attack...' : 'Run FGSM Attack'}
            </button>
            {error && <p className="text-red-400 mt-4 bg-red-900/30 p-2 rounded">{error}</p>}
          </div>
        </div>

        {/* Results Display */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          
          {/* Original Image Card */}
          <div className="bg-gray-800 p-4 rounded-xl border border-gray-700 flex flex-col items-center">
            <h2 className="text-xl font-semibold mb-4 text-green-400">Original Image</h2>
            {preview ? (
              <img src={preview} alt="Original" className="w-48 h-48 object-contain bg-black rounded-lg border border-gray-600" />
            ) : (
              <div className="w-48 h-48 flex items-center justify-center bg-gray-700 rounded-lg text-gray-500">No Image</div>
            )}
            <div className="mt-4 text-center">
              <p className="text-sm text-gray-400">Model Prediction:</p>
              <p className="text-3xl font-bold text-white">
                {result ? result.clean_prediction : '-'}
              </p>
            </div>
          </div>

          {/* Adversarial Image Card */}
          <div className="bg-gray-800 p-4 rounded-xl border border-gray-700 flex flex-col items-center relative overflow-hidden">
            {result?.success && (
              <div className="absolute top-0 right-0 bg-red-600 text-white text-xs font-bold px-3 py-1 rounded-bl-lg">
                ATTACK SUCCESS
              </div>
            )}
            <h2 className="text-xl font-semibold mb-4 text-red-400">Adversarial Image</h2>
            {result ? (
              <img src={`data:image/png;base64,${result.adversarial_image}`} alt="Adversarial" className="w-48 h-48 object-contain bg-black rounded-lg border border-red-900/50" />
            ) : (
              <div className="w-48 h-48 flex items-center justify-center bg-gray-700 rounded-lg text-gray-500">Run Attack First</div>
            )}
            <div className="mt-4 text-center">
              <p className="text-sm text-gray-400">Model Prediction:</p>
              <p className="text-3xl font-bold text-white">
                {result ? result.adversarial_prediction : '-'}
              </p>
            </div>
          </div>

        </div>
      </div>
    </main>
  );
}