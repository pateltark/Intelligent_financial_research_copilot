import { NextResponse } from "next/server";

export function middleware(request) {
    const token = request.cookies.get("token")?.value;
    const { pathname } = request.nextUrl;

    if (pathname.startsWith("/chat") && !token) {
        return NextResponse.redirect(new URL("/login", request.url));
    }

    if (pathname === "/") {
        return NextResponse.redirect(new URL("/login", request.url));
    }

    return NextResponse.next();
}

export const config = {
    matcher: ["/", "/chat/:path*"],
};