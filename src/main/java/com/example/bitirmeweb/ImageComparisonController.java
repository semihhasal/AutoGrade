package com.example.bitirmeweb;

import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;

@Controller
public class ImageComparisonController {

    @GetMapping("/")
    public String home(Model model) {
        return "index"; // index.html sayfasına yönlendirir.
    }
}