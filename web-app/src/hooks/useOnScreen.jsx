import { useState, useEffect, useRef } from 'react';

export default function useOnScreen(options) {
    const ref = useRef(null);
    const [isIntersecting, setIntersecting] = useState(false);

    const { threshold = 0.1, triggerOnce = true } = options || {};

    useEffect(() => {
        const observer = new IntersectionObserver(([entry]) => {
            // If it should trigger every time, update state on every intersection change.
            if (!triggerOnce) {
                setIntersecting(entry.isIntersecting);
                return;
            }
            // If it should trigger only once, set state to true and then unobserve.
            if (entry.isIntersecting) {
                setIntersecting(true);
                observer.unobserve(entry.target);
            }
        }, { threshold });

        const currentRef = ref.current;
        if (currentRef) {
            observer.observe(currentRef);
        }

        return () => {
            if (currentRef) {
                observer.unobserve(currentRef);
            }
        };
    }, [ref, threshold, triggerOnce]);

    return [ref, isIntersecting];
}
