import React, { useState } from 'react';
import SearchableDropdown from './SearchableDropdown';
import InfoCard from './InfoCard';
import ConditionTag from './ConditionTag';

const DetailView = ({ title, body }) => (
    <div>
        <h3 className="text-xl font-bold text-gray-800 mb-2">{title}</h3>
        <p className="text-gray-600">{body}</p>
    </div>
);

const PatientHistoryColumn = ({ selectedPatient, patients, handlePatientSelect, setModalContent }) => {
    const [isDropdownOpen, setDropdownOpen] = useState(false);
    if (!selectedPatient) return <div className="w-1/3 p-4">Loading...</div>;
    return (
        <div className="w-1/3 bg-gray-50 p-4 rounded-lg shadow-inner overflow-y-auto border-r border-gray-200">
            <div className="mb-4 relative">
                 <button onClick={() => setDropdownOpen(!isDropdownOpen)} className="w-full flex justify-between items-center p-2 rounded-md hover:bg-gray-200 transition-colors">
                    <div>
                        <h2 className="text-xl font-bold text-left" style={{color: '#2C3E50'}}>{selectedPatient.name}</h2>
                        <p className="text-sm text-gray-600 text-left">{selectedPatient.age}Y {selectedPatient.gender} • {selectedPatient.bloodGroup}</p>
                    </div>
                    <span className={`transform transition-transform ${isDropdownOpen ? 'rotate-180' : ''}`}>▼</span>
                 </button>
                 {isDropdownOpen && ( <SearchableDropdown options={patients} onSelect={handlePatientSelect} onClose={() => setDropdownOpen(false)} /> )}
            </div>
            <div className="bg-teal-50 border border-teal-200 text-teal-900 text-xs p-3 rounded-md mb-4">{selectedPatient.aiSummary}</div>
            <div className="space-y-3">
                 <InfoCard title="Red Flag Alerts"><ConditionTag text="Penicillin Allergy" color="red" onClick={() => setModalContent(<DetailView title="Penicillin Allergy Details" body="Reported on 15 Mar 2025 following a mild skin rash." />)} /></InfoCard>
                 <InfoCard title="Pre-existing Conditions">{selectedPatient.preExistingConditions.map(c => <ConditionTag key={c} text={c} color="orange" onClick={() => setModalContent(<DetailView title={`History of ${c}`} body="Condition first diagnosed on..." />)} />)}</InfoCard>
                 <InfoCard title="Known Allergies">{selectedPatient.allergies.filter(a => a !== 'Penicillin').map(a => <ConditionTag key={a} text={a} color="orange" />)}</InfoCard>
                 <InfoCard title="Recurring Illnesses">{selectedPatient.recurringIllnesses.map(i => <ConditionTag key={i} text={i} color="blue" />)}</InfoCard>
                 <InfoCard title="Drug-Drug Interactions">{selectedPatient.drugInteractions.length > 0 ? selectedPatient.drugInteractions.map(d => <ConditionTag key={d.drugA} text={`${d.drugA} + ${d.drugB}`} color="red" />) : <span className="text-xs text-gray-500">None detected.</span>}</InfoCard>
            </div>
        </div>
    );
};

export default PatientHistoryColumn;
