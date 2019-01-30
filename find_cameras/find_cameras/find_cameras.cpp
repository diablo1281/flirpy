// find_cameras.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <windows.h>
#include <dshow.h>
#include <string>
#include <iostream>

#pragma comment(lib, "strmiids")


HRESULT EnumerateDevices(REFGUID category, IEnumMoniker **ppEnum)
{
	// Create the System Device Enumerator.
	ICreateDevEnum *pDevEnum;
	HRESULT hr = CoCreateInstance(CLSID_SystemDeviceEnum, NULL,
		CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&pDevEnum));

	if (SUCCEEDED(hr))
	{
		// Create an enumerator for the category.
		hr = pDevEnum->CreateClassEnumerator(category, ppEnum, 0);
		if (hr == S_FALSE)
		{
			hr = VFW_E_NOT_FOUND;  // The category is empty. Treat as an error.
		}
		pDevEnum->Release();
	}
	return hr;
}

int get_id(IEnumMoniker *pEnum, std::string friendly_name)
{
	IMoniker *pMoniker = NULL;
	int i = 0;

	while (pEnum->Next(1, &pMoniker, NULL) == S_OK)
	{
		IPropertyBag *pPropBag;
		HRESULT hr = pMoniker->BindToStorage(0, 0, IID_PPV_ARGS(&pPropBag));
		if (FAILED(hr))
		{
			pMoniker->Release();
			continue;
		}

		VARIANT var;
		VariantInit(&var);

		// Get description or friendly name.
		hr = pPropBag->Read(L"Description", &var, 0);
		if (FAILED(hr))
		{
			hr = pPropBag->Read(L"FriendlyName", &var, 0);
		}
		if (SUCCEEDED(hr))
		{
			std::wstring str(var.bstrVal);

			if (std::string(str.begin(), str.end()) == friendly_name) {
				hr = pPropBag->Read(L"DevicePath", &var, 0);
				return i;
			}
			VariantClear(&var);
		}

		i++;

		pPropBag->Release();
		pMoniker->Release();
	}

	return -1;
}

int find_device(std::string friendly_name) {
	HRESULT hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
	int id = -1;
	if (SUCCEEDED(hr))
	{
		IEnumMoniker *pEnum;

		hr = EnumerateDevices(CLSID_VideoInputDeviceCategory, &pEnum);
		if (SUCCEEDED(hr))
		{
			id = get_id(pEnum, friendly_name);
			pEnum->Release();
		}
		CoUninitialize();
	}
	return id;
}

int main(int argc, char** argv)
{
	if (argc == 2) {
		std::cout << find_device(argv[1]);
	}
	
	return 0;
}