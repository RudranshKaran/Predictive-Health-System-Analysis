import React, { useRef, useEffect } from 'react';

const DataWeaveAnimation = () => {
    const canvasRef = useRef(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        let width, height, time = 0;
        const waveCount = 10;
        const waves = [];

        function setup() {
            width = canvas.width = window.innerWidth;
            height = canvas.height = window.innerHeight;
            waves.length = 0;
            for (let i = 0; i < waveCount; i++) {
                waves.push({
                    y: Math.random() * height,
                    length: Math.random() * 0.02 + 0.005,
                    amplitude: Math.random() * 50 + 20,
                    speed: Math.random() * 0.01 + 0.005,
                    // The user requested 0.12 opacity, so I'll adjust this calculation
                    // to be centered around that value while keeping some randomness.
                    opacity: Math.random() * 0.13 + 0.07 // Ranges from 0.07 to 0.17
                });
            }
        }

        let animationFrameId;
        function draw() {
            ctx.clearRect(0, 0, width, height);
            time += 0.5;

            waves.forEach(wave => {
                ctx.beginPath();
                // Using theme's teal color: rgba(38, 166, 154, opacity)
                ctx.strokeStyle = `rgba(38, 166, 154, ${wave.opacity})`;
                ctx.lineWidth = 1;

                for (let x = 0; x < width; x++) {
                    const y = wave.y + Math.sin(x * wave.length + time * wave.speed) * wave.amplitude;
                    if (x === 0) {
                        ctx.moveTo(x, y);
                    } else {
                        ctx.lineTo(x, y);
                    }
                }
                ctx.stroke();
            });

            animationFrameId = requestAnimationFrame(draw);
        }

        window.addEventListener('resize', setup);
        setup();
        draw();

        return () => {
            window.removeEventListener('resize', setup);
            cancelAnimationFrame(animationFrameId);
        };
    }, []);

    return <canvas ref={canvasRef} style={{ position: 'fixed', top: 0, left: 0, width: '100%', height: '100%', zIndex: -1, background: '#ffffff' }} />;
};

export default DataWeaveAnimation;
