import React from 'react';

const MicIcon = ({ isListening }) => <svg xmlns="http://www.w3.org/2000/svg" className={`h-6 w-6 ${isListening ? 'text-red-500' : 'text-gray-500'}`} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-14 0m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" /></svg>;

export default MicIcon;
