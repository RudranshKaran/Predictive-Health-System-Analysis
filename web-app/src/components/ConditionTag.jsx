import React from 'react';

const ConditionTag = ({ text, color, onClick }) => {
    const colorClasses = { red: 'bg-red-100 text-red-800', orange: 'bg-orange-100 text-orange-800', blue: 'bg-blue-100 text-blue-800', green: 'bg-green-100 text-green-800', yellow: 'bg-yellow-100 text-yellow-800', teal: 'bg-teal-100 text-teal-800' };
    const Tag = onClick ? 'button' : 'span';
    return ( <Tag onClick={onClick} className={`px-2.5 py-1 text-xs font-semibold rounded-full transition-transform ${colorClasses[color]} ${onClick ? 'hover:scale-105' : ''}`}> {text} </Tag> );
};

export default ConditionTag;
