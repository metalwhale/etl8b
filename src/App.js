import React, { useRef, useEffect, useState } from 'react';
import SignatureCanvas from 'react-signature-canvas';
import * as tf from '@tensorflow/tfjs';
import './App.css';
import labels from './labels';

function App() {
  const IMAGE_SIZE = 256;
  const INPUT_SHAPE = [64, 64];
  const PROBABILITY_THRESHOLD = 0.01;
  const signature = useRef(null);
  const [model, setModel] = useState(null);
  const [matches, setMatches] = useState([]);

  useEffect(() => {
    tf.loadLayersModel('/model/model.json').then(model => {
      setModel(model);
    });
  }, []);

  function clear() {
    signature.current.clear();
  }

  function recognize() {
    const canvas = signature.current.getTrimmedCanvas();
    const context = canvas.getContext('2d');
    const data = context.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE);
    const image = tf.browser.fromPixels(data, 1).resizeBilinear(INPUT_SHAPE).expandDims().minimum(tf.scalar(1));
    model.predict(image, {batchSize: 1}).data().then(results => {
      const predictions = results.reduce((p, r, i) => {
        if (r >= PROBABILITY_THRESHOLD) {
          p.push({character: labels[i], probability: r});
        }
        return p;
      }, []);
      predictions.sort((p1, p2) => p2.probability - p1.probability);
      setMatches(predictions);
    });
  }

  return (
    <div className="app">
      <div className="container">
        <SignatureCanvas
          ref={signature}
          backgroundColor='black'
          penColor='white'
          canvasProps={{width: IMAGE_SIZE, height: IMAGE_SIZE, className: 'signature-canvas'}}
        />
        {model ? (
          <div className="clearfix">
            <button onClick={clear} style={{float: "left"}}>Clear</button>
            <button onClick={recognize} style={{float: "right"}}>Recognize</button>
          </div>
        ) : (
          <div>Loading model...</div>
        )}
        <div>
          {matches.map((match, index) => (
            <div key={index}>{match.character} ({Math.round(match.probability * 100)}%)</div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default App;
