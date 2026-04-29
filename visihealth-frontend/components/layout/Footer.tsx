"use client";

import React from "react";
import Link from "next/link";
import { Github, Mail, Heart } from "lucide-react";

export function Footer() {
  return (
    <footer className="bg-gray-900 text-white">
      <div className="mx-auto max-w-7xl px-4 py-12 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {/* Brand */}
          <div>
            <h3 className="text-lg font-bold mb-4">VisiHealth AI</h3>
            <p className="text-gray-400 text-sm">
              AI-Powered Medical Visual Question Answering System
            </p>
            <p className="text-gray-400 text-sm mt-2">
              Built with ResNet50 + BioLinkBERT
            </p>
          </div>

          {/* Links */}
          <div>
            <h3 className="text-lg font-semibold mb-4">Quick Links</h3>
            <ul className="space-y-2">
              <li>
                <Link href="/" className="text-gray-400 hover:text-white transition-colors text-sm">
                  Home
                </Link>
              </li>
              <li>
                <Link href="/analyze" className="text-gray-400 hover:text-white transition-colors text-sm">
                  Analyze
                </Link>
              </li>
              <li>
                <Link href="/history" className="text-gray-400 hover:text-white transition-colors text-sm">
                  History
                </Link>
              </li>
              <li>
                <Link href="/about" className="text-gray-400 hover:text-white transition-colors text-sm">
                  About
                </Link>
              </li>
            </ul>
          </div>

          {/* Contact */}
          <div>
            <h3 className="text-lg font-semibold mb-4">Connect</h3>
            <div className="flex space-x-4">
              <a
                href="https://github.com"
                target="_blank"
                rel="noopener noreferrer"
                className="text-gray-400 hover:text-white transition-colors"
              >
                <Github className="h-6 w-6" />
              </a>
              <a
                href="mailto:contact@visihealth.ai"
                className="text-gray-400 hover:text-white transition-colors"
              >
                <Mail className="h-6 w-6" />
              </a>
            </div>
          </div>
        </div>

        {/* Bottom */}
        <div className="mt-8 pt-8 border-t border-gray-800">
          <p className="text-center text-gray-400 text-sm flex items-center justify-center">
            © 2026 VisiHealth AI. Built with <Heart className="h-4 w-4 mx-1 text-red-500" /> for
            better healthcare.
          </p>
        </div>
      </div>
    </footer>
  );
}
