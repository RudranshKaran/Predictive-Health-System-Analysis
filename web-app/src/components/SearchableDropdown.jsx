import React, { useState, useEffect, useRef } from 'react';

const SearchableDropdown = ({ options, onSelect, onClose }) => {
    const [searchTerm, setSearchTerm] = useState('');
    const dropdownRef = useRef(null);
    const filteredOptions = options.filter(option => option.name.toLowerCase().includes(searchTerm.toLowerCase()));

    useEffect(() => {
        const handleClickOutside = (event) => { if (dropdownRef.current && !dropdownRef.current.contains(event.target)) { onClose(); } };
        document.addEventListener("mousedown", handleClickOutside);
        return () => document.removeEventListener("mousedown", handleClickOutside);
    }, [dropdownRef, onClose]);
    
    return (
      <div ref={dropdownRef} className="absolute top-full mt-2 w-full bg-white rounded-md shadow-lg border border-gray-200 z-50">
        <div className="p-2">
          <input type="text" placeholder="Search patient..." className="w-full px-3 py-2 text-sm border border-gray-300 rounded-md focus:ring-1 focus:ring-teal-500" value={searchTerm} onChange={(e) => setSearchTerm(e.target.value)} autoFocus />
        </div>
        <ul className="max-h-60 overflow-y-auto">
          {filteredOptions.length > 0 ? filteredOptions.map(option => (
            <li key={option.id}>
              <button onClick={() => { onSelect(option.id); onClose(); }} className="w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-teal-50"> {option.name} <span className="text-gray-400">({option.age}{option.gender})</span> </button>
            </li>
          )) : ( <li className="px-4 py-2 text-sm text-gray-500">No patients found.</li> )}
        </ul>
      </div>
    );
};

export default SearchableDropdown;
