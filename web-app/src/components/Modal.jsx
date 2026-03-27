import React from 'react';

const Modal = ({ content, onClose }) => {
    return (
        <div 
            className="fixed inset-0 z-50 flex items-center justify-center" 
            style={{ backgroundColor: 'rgba(0, 0, 0, 0.3)', backdropFilter: 'blur(8px)', WebkitBackdropFilter: 'blur(8px)' }}
            onClick={onClose}
        >
            <div 
                className="bg-white rounded-lg shadow-2xl w-full max-w-lg p-6 relative"
                onClick={e => e.stopPropagation()} // Prevent click from closing modal
            >
                <button 
                    onClick={onClose} 
                    className="absolute top-3 right-3 text-gray-400 hover:text-gray-600"
                    aria-label="Close modal"
                >
                    <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                </button>
                {content}
            </div>
        </div>
    );
};

export default Modal;
