import React from 'react';
import { medicationData } from '../data/medicationData';

const PillIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-3 text-teal-600 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
);

const MedicationDetail = ({ medicationName }) => {
    const data = medicationData[medicationName];
    if (!data) {
        return <div>Details not found for {medicationName}.</div>;
    }

    return (
        <div>
            <h3 className="text-xl font-bold text-gray-800 mb-2">{medicationName}</h3>
            <p className="text-gray-600 mb-4">{data.description}</p>
            <h4 className="font-semibold text-gray-700 mb-2">Potential Interactions:</h4>
            <ul className="list-disc list-inside space-y-1">
                {data.interactions.map(interaction => (
                    <li key={interaction} className="text-gray-600">
                        <span className="font-medium text-gray-800">{interaction}:</span> {data.interactionDetails[interaction]}
                    </li>
                ))}
            </ul>
        </div>
    );
};


const ActiveMedications = ({ medications, setModalContent }) => {
    const handleMedicationClick = (medicationName) => {
        setModalContent(<MedicationDetail medicationName={medicationName} />);
    };
    
    if (!medications || medications.length === 0) {
        return (
            <div className="bg-white p-6 rounded-lg shadow-md h-full">
                <h3 className="text-lg font-semibold text-gray-800 mb-4">Active Medications</h3>
                <div className="flex items-center justify-center h-full text-gray-500">
                    <p>No active medications reported.</p>
                </div>
            </div>
        );
    }

    return (
        <div className="bg-white p-6 rounded-lg shadow-md h-full">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">Active Medications</h3>
            <div className="space-y-3 overflow-y-auto h-[calc(100%-2rem)] pr-2">
                {medications.map((medication, index) => (
                    <button 
                        key={index} 
                        className="flex items-center bg-teal-50 p-3 rounded-md border border-teal-200 hover:shadow-md transition-shadow w-full text-left"
                        onClick={() => handleMedicationClick(medication)}
                    >
                        <PillIcon />
                        <span className="text-teal-900 font-medium">{medication}</span>
                    </button>
                ))}
            </div>
        </div>
    );
};

export default ActiveMedications;
